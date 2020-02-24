from unet import custom_unet
import os
import horovod.tensorflow as hvd
import tensorflow as tf
from utils_logger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
import argparse
from data_loader import Dataset
#from run_v2 import train,restore_checkpoint # no mixed_precision PURE horovod 
from run import train, restore_checkpoint



def main(params):
    """
    Starting point of the application
    """
    backends = [StdOutBackend(Verbosity.VERBOSE)]
    if params.log_dir is not None:
        os.makedirs(params.log_dir, exist_ok=True)
        logfile = os.path.join(params.log_dir, "log.json")
        backends.append(JSONStreamBackend(Verbosity.VERBOSE, logfile))
    logger = Logger(backends)

    # Optimization flags
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

    hvd.init()
    #set parameters

    if params.use_xla:
        tf.config.optimizer.set_jit(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    if params.use_amp:
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    else:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
    
    # get dataset from tf.data api
    dataset = Dataset(batch_size=params.batch_size,gpu_id=hvd.rank(),num_gpus=hvd.size())

    
    # Build the  model
    input_shape=(1024,2048,3)
    model=custom_unet(input_shape,
        num_classes=8,
        use_batch_norm=False, 
        upsample_mode='decov', # 'deconv' or 'simple' 
        use_dropout_on_upsampling=True, 
        dropout=0.3, 
        dropout_change_per_layer=0.0,
        filters=7,
        num_layers=4,
        output_activation='softmax')
    #model.compile(optimizer=opt, loss=combined_dice_binary_loss, metrics=[dice_coef],experimental_run_tf_function=False)

    #start training 
    train(params, model, dataset, logger)

if __name__ == '__main__':


    PARSER = argparse.ArgumentParser(description="Unet_Cityscape")

    PARSER.add_argument('--exec_mode',
                        choices=['train', 'train_and_predict', 'predict', 'evaluate', 'train_and_evaluate'],
                        type=str,
                        default='train',
                        help="""Execution mode of running the model""")

    PARSER.add_argument('--model_dir',
                        type=str,
                        default='./checkpt',
                        help="""Output directory for information related to the model""")

    PARSER.add_argument('--log_dir',
                        type=str,
                        default=None,
                        help="""Output directory for training logs""")

    PARSER.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help="""Size of each minibatch per GPU""")

    PARSER.add_argument('--learning_rate',
                        type=float,
                        default=0.0001,
                        help="""Learning rate coefficient for AdamOptimizer""")

    PARSER.add_argument('--max_steps',
                        type=int,
                        default=100,
                        help="""Maximum number of steps (batches) used for training""")

    PARSER.add_argument('--weight_decay',
                        type=float,
                        default=0.0005,
                        help="""Weight decay coefficient""")

    PARSER.add_argument('--log_every',
                        type=int,
                        default=100,
                        help="""Log performance every n steps""")

    PARSER.add_argument('--warmup_steps',
                        type=int,
                        default=0,
                        help="""Number of warmup steps""")

    PARSER.add_argument('--seed',
                        type=int,
                        default=0,
                        help="""Random seed""")

    PARSER.add_argument('--benchmark', dest='benchmark', action='store_true',
                        help="""Collect performance metrics during training""")
    PARSER.add_argument('--no-benchmark', dest='benchmark', action='store_false')
    PARSER.set_defaults(augment=False)

    PARSER.add_argument('--use_amp', dest='use_amp', action='store_true',
                        help="""Train using TF-AMP""")
    PARSER.set_defaults(use_amp=False)

    PARSER.add_argument('--use_xla', dest='use_xla', action='store_true',
                        help="""Train using XLA""")
    PARSER.set_defaults(use_amp=False)

    PARSER.add_argument('--use_trt', dest='use_trt', action='store_true',
                        help="""Use TF-TRT""")
    PARSER.set_defaults(use_trt=False)
    params=PARSER.parse_args()

    main(params)
