import os 
import numpy as np
import tensorflow as tf
print(tf.__version__)
tf.debugging.set_log_device_placement(True)
#%%writefile  test_cprofile_example.py
# url from :https://www.tensorflow.org/tutorials/load_data/images
# how to read the output - https://ymichael.com/2014/03/08/profiling-python-with-cprofile.html

print(tf.__version__)
tf.executing_eagerly()

from hvd.unet import custom_unet
from loss_metrics import dice_loss, combined_dice_binary_loss , dice_coef
import time
batch_size=4
@profile
def one_hot(image, gt):
    gt_shape = tf.shape(gt)
    gt = tf.reshape(gt, gt_shape[:-1])
    gt = tf.one_hot(gt, 8)
    return image, gt
@profile
def decode_img(colored_image,ch=3):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(colored_image, channels=ch)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.

    return img
@profile
def process_path(img_file_path):
    mask_file_path=tf.strings.regex_replace(img_file_path,'img','gt_cat')
    mask_file_path=tf.strings.regex_replace(mask_file_path,'_leftImg8bit.png', '_gtFine_color.png')# '_gtFine_labelIds.png')
    #print(img_file_path,mask_file_path)
    img_file = tf.io.read_file(img_file_path)
    mask_file = tf.io.read_file(mask_file_path)
    return img_file, mask_file
@profile
def decode(img_file, mask_file):
    #gt_file=tf.io.read_file(mask_file_path)
    im=decode_img(img_file,ch=3)
    gt=decode_img(mask_file,ch=1)
    print(im.shape, gt.shape)
    gt=tf.cast(gt,tf.uint8)
    return im,gt
@profile
def adapt_types_and_transpose(im, gt):
    im = tf.cast(im, tf.float32)
    #im = tf.transpose(im , (2,0,1)) # make sure that the input images are with the shape = (N_batches , Channels , Height, Width) for model to eat
    gt = tf.cast(gt, tf.uint8)
    return im, gt

@profile
def tf_data_pipeline_fn():
    img_files = tf.data.Dataset.list_files('/workspace/8data/img/*.png')
    dataset=img_files.map(lambda x: process_path(x)) 
    dataset=dataset.map(decode)
    # adapt the data types
    dataset = dataset.map(adapt_types_and_transpose)
    dataset = dataset.map(one_hot)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=128)
    
    dataset = dataset.prefetch(buffer_size=128)
    return dataset
@profile
def run_tf2strategy():
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    NUM_WORKERS = 8 # number of gpus used for training
    # Here the batch size scales up by number of workers , previously we got 4 as base_batch_size, now after scale by the num_gpus, we got 4*8=32
    batch_size = 4 * NUM_WORKERS 
    
    # configure to use mixed precison training
    USE_AMP = True
    USE_XLA = True
    tf.config.optimizer.set_jit(USE_XLA)
    # Load Dataset
    # This `fit` call will be distributed on 8 GPUs.
 
    
    devices = ['/device:GPU:{}'.format(i) for i in range(NUM_WORKERS)] # make sure you pass the available devices into strategy , otherwise Assertion error will arise
    strategy = tf.distribute.MirroredStrategy(devices) # this is the strategy for 1 node but multiple gpus
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    startTime=time.time()
    if USE_AMP:
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    with strategy.scope():
        dataset = tf_data_pipeline_fn().batch(batch_size) # make batch based on global batch_size
        opt = tf.optimizers.Adam(1e-4) 
        input_shape=(1024,2048,3)
        parallel_model=custom_unet(input_shape,
            num_classes=8,
            use_batch_norm=False, 
            upsample_mode='simple', # 'deconv' or 'simple' 
            use_dropout_on_upsampling=True, 
            dropout=0.3, 
            dropout_change_per_layer=0.0,
            filters=7,
            num_layers=4,
            output_activation='softmax')
        parallel_model.compile(optimizer=opt, loss=combined_dice_binary_loss, metrics=[dice_coef],experimental_run_tf_function=False)
    # directly call the fit function
    
    parallel_model.fit(dataset, epochs=1,steps_per_epoch=1)
    print('Using 8 GPUs in parallel for TF2 strategy training and add MixedPrecision took {0} seconds'.format(time.time() - startTime))
    parallel_model.save_weights('checkpt/8gpu1node')

if __name__ == "__main__":
    run_tf2strategy()
    






