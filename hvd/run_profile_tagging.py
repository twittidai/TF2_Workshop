import os
from time import time
import numpy as np
import horovod.tensorflow as hvd
import tensorflow as tf


# Class Dice coefficient averaged over batch
def dice_coef(predict, target, axis=1, eps=1e-6):
    intersection = tf.reduce_sum(input_tensor=predict * target, axis=axis)
    union = tf.reduce_sum(input_tensor=predict * predict + target * target, axis=axis)
    dice = (2. * intersection + eps) / (union + eps)
    return tf.reduce_mean(input_tensor=dice, axis=0)  # average over batch


def partial_losses(predict, target):
    n_classes = predict.shape[-1]

    flat_logits = tf.reshape(tf.cast(predict, tf.float32),
                             [tf.shape(input=predict)[0], -1, n_classes])
    flat_labels = tf.reshape(target,
                             [tf.shape(input=predict)[0], -1, n_classes])

    crossentropy_loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                                            labels=flat_labels),
                                       name='cross_loss_ref')
    dice_loss = tf.reduce_mean(input_tensor=1 - dice_coef(flat_logits, flat_labels), name='dice_loss_ref')
    return crossentropy_loss, dice_loss


def process_performance_stats(timestamps, params):
    warmup_steps = params.warmup_steps
    batch_size = params.batch_size
    timestamps_ms = 1000 * timestamps[warmup_steps:] / batch_size
    timestamps_ms = timestamps_ms[timestamps_ms > 0]
    latency_ms = timestamps_ms.mean()
    std = timestamps_ms.std()
    n = np.sqrt(len(timestamps_ms))
    throughput_imgps = (1000.0 / timestamps_ms).mean()
    print('Throughput Avg:', round(throughput_imgps, 3), 'img/s')
    print('Latency Avg:', round(latency_ms, 3), 'ms')
    for ci, lvl in zip(["90%:", "95%:", "99%:"],
                       [1.645, 1.960, 2.576]):
        print("Latency", ci, round(latency_ms + lvl * std / n, 3), "ms")
    return float(throughput_imgps), float(latency_ms)



def restore_checkpoint(model, model_dir):
    try:
        model.load_weights(os.path.join(model_dir, "checkpoint"))
    except:
        print("Failed to load checkpoint, model will have randomly initialized weights.")
    return model

@profile
def train(params, model, dataset, logger):
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)
    max_steps = params.max_steps // hvd.size()

    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    if params.use_amp:
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

    ce_loss = tf.keras.metrics.Mean(name='ce_loss')
    f1_loss = tf.keras.metrics.Mean(name='dice_loss')
    @profile
    @tf.function
    def train_step(features, labels, warmup_batch=False):
        with tf.GradientTape() as tape:
            output_map = model(features)
            crossentropy_loss, dice_loss = partial_losses(output_map, labels)
            added_losses = tf.add(crossentropy_loss, dice_loss, name="total_loss_ref")
            loss = added_losses + params.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in model.trainable_variables
                 if 'batch_normalization' not in v.name])

            if params.use_amp:
                loss = optimizer.get_scaled_loss(loss)
        tape = hvd.DistributedGradientTape(tape)
        gradients = tape.gradient(loss, model.trainable_variables)
        if params.use_amp:
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if warmup_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        ce_loss(crossentropy_loss)
        f1_loss(dice_loss)



    if params.benchmark:
        assert max_steps * hvd.size() > params.warmup_steps, \
            "max_steps value has to be greater than warmup_steps"
        timestamps = np.zeros((hvd.size(), max_steps * hvd.size() + 1), dtype=np.float32)
        for iteration, (images, labels) in enumerate(dataset.train_fn(drop_remainder=True)):
            t0 = time()
            train_step(images, labels, warmup_batch=iteration == 0)
            timestamps[hvd.rank(), iteration] = time() - t0
            if iteration >= max_steps * hvd.size():
                break
        timestamps = np.mean(timestamps, axis=0)
        throughput_imgps, latency_ms = process_performance_stats(timestamps, params)
        if hvd.rank() == 0:
            logger.log(step=(),
                       data={"average_images_per_second": throughput_imgps,
                             "average_latency": latency_ms})
    else:
        for iteration, (images, labels) in enumerate(dataset.train_fn(drop_remainder=True)):
            train_step(images, labels, warmup_batch=iteration == 0)
            if (hvd.rank() == 0) and (iteration % params.log_every == 0):
                logger.log(step=(iteration, max_steps),
                           data={"train_ce_loss": float(ce_loss.result()),
                                 "train_dice_loss": float(f1_loss.result()),
                                 "train_total_loss": float(f1_loss.result() + ce_loss.result())})

            f1_loss.reset_states()
            ce_loss.reset_states()

            if iteration >= max_steps:
                break
        if hvd.rank() == 0:
            model.save_weights(os.path.join(params.model_dir, "checkpoint"))
    logger.flush()


def evaluate(params, model, dataset, logger):
    ce_loss = tf.keras.metrics.Mean(name='ce_loss')
    f1_loss = tf.keras.metrics.Mean(name='dice_loss')

    @tf.function
    def validation_step(features, labels):
        output_map = model(features, is_training=False)
        crossentropy_loss, dice_loss = partial_losses(output_map, labels)
        ce_loss(crossentropy_loss)
        f1_loss(dice_loss)

    for iteration, (images, labels) in enumerate(dataset.train_fn(drop_remainder=True)):
        validation_step(images, labels)
        if iteration >= dataset.eval_size // params.batch_size:
            break
    if dataset.eval_size > 0:
        logger.log(step=(),
                   data={"eval_ce_loss": float(ce_loss.result()),
                         "eval_dice_loss": float(f1_loss.result()),
                         "eval_total_loss": float(f1_loss.result() + ce_loss.result()),
                         "eval_dice_score": 1.0 - float(f1_loss.result())})

    logger.flush()
