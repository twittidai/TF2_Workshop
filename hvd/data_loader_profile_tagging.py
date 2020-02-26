import os
from time import time
import numpy as np
import horovod.tensorflow as hvd
import tensorflow as tf
import os
import numpy as np
import multiprocessing
##### all for constructing tf.data dataset


class Dataset:
    """Load, separate and prepare the data for training and prediction"""
    def __init__(self, batch_size, gpu_id=0, num_gpus=1):
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.gpu_id = gpu_id
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.img_files = tf.data.Dataset.list_files('/workspace/8data/img/*.png')
        self.dataset=self.img_files.map(lambda x: self._process_path(x)) 
        self.dataset = self.dataset.shard(self.num_gpus, self.gpu_id)
        self.dataset = self.dataset.map(self._load_img,num_parallel_calls=multiprocessing.cpu_count()//self.num_gpus)
        # adapt the data types
        self.dataset = self.dataset.map(self._adapt_types_and_transpose,num_parallel_calls=multiprocessing.cpu_count()//self.num_gpus)
        self.dataset = self.dataset.map(self._one_hot,num_parallel_calls=multiprocessing.cpu_count()//self.num_gpus)
        print("the global batch size is ", str(batch_size*num_gpus))
        
    @profile
    def _one_hot(self,image, gt):
        gt_shape = tf.shape(gt)
        gt = tf.reshape(gt, gt_shape[:-1])
        gt = tf.one_hot(gt, 8)
        return image, gt
    @profile
    def _decode_img(self,colored_image,ch=3):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(colored_image, channels=ch)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.

        return img
    @profile
    def _load_img(self,img_file,mask_file):
        #gt_file=tf.io.read_file(mask_file_path)
        im=self._decode_img(img_file, ch=3)
        gt=self._decode_img(mask_file,ch=1)
        print(im.shape, gt.shape)
        return im,gt 
    @profile
    def _process_path(self,img_file_path):
        mask_file_path=tf.strings.regex_replace(img_file_path,'img','gt_cat')
        mask_file_path=tf.strings.regex_replace(mask_file_path,'_leftImg8bit.png', '_gtFine_color.png')
        
        #print(img_file_path,mask_file_path)
        img_file = tf.io.read_file(img_file_path)
        mask_file = tf.io.read_file(mask_file_path)
        return img_file, mask_file

    @profile
    def _adapt_types_and_transpose(self,im, gt):
        im = tf.cast(im, tf.float32)
        # cast gt to uint8 otherwise one hot encoding wont work
        gt = tf.cast(gt, tf.uint8)
        return im, gt

    ### consolidating all manipulation into one train_fn 
    @profile
    def train_fn(self,drop_remainder=False):
        """Input function for training"""
        #self.dataset = self.dataset.shard(self.num_gpus, self.gpu_id)
        self.dataset = self.dataset.repeat() # repeat 
        self.dataset = self.dataset.shuffle(buffer_size=128) # shuffle and give a buffer size
        self.dataset = self.dataset.batch(self.batch_size) # make batch
        # `prefetch` lets the dataset fetch batches in the background while the model
        self.dataset = self.dataset.prefetch(buffer_size=self.AUTOTUNE)  
        return self.dataset

