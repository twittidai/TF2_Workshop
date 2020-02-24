import os 
import numpy as np
import cv2
import tensorflow as tf
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
def tf_data_pipeline_fn(batch_size=4):
    img_files = tf.data.Dataset.list_files('/workspace/8data/img/*.png')
    dataset=img_files.map(lambda x: process_path(x)) 
    dataset=dataset.map(decode)
    # adapt the data types
    dataset = dataset.map(adapt_types_and_transpose)
    dataset = dataset.map(one_hot)
    #dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=128)
    
    dataset = dataset.prefetch(buffer_size=128)
    dataset = dataset.batch(batch_size)
    return dataset

if __name__ == "__main__":
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ##### print it to ensure tf.data pipeline works  to read in the pair of image, mask files by filename 
    dataset = tf_data_pipeline_fn()
    i=0
    for im,ma in dataset:
        # print and ensure the data and mask shape are correct, i.e for image=( 1024,2048 ,3) and for mask=(1024,2048,8) 
        #print(im.shape, type(ma),ma.shape)
        i+=1
    print("==============finished===========")