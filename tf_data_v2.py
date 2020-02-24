import os 
import numpy as np
import tensorflow as tf
import time

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
def load_img(img_file,mask_file):
    #gt_file=tf.io.read_file(mask_file_path)
    im=decode_img(img_file, ch=3)
    gt=decode_img(mask_file,ch=1)
    print(im.shape, gt.shape)
    return im,gt 
@profile
def read_in_files(img_file_path,mask_file_path):
    #print(img_file_path,mask_file_path)
    img_file = tf.io.read_file(img_file_path)
    mask_file = tf.io.read_file(mask_file_path)
    return img_file, mask_file
@profile
def adapt_types_and_transpose(im, gt):
    im = tf.cast(im, tf.float32)
    gt = tf.cast(gt, tf.uint8)
    return im, gt

@profile
def tf_data_pipeline_fn(batch_size=4):
    img_files=tf.data.TextLineDataset('./img8_small.txt')
    mask_files=tf.data.TextLineDataset('./gt_cat8_small.txt')
    dataset = tf.data.Dataset.zip((img_files,mask_files))
    dataset = dataset.map(read_in_files)
    dataset = dataset.map(load_img)
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
    print("===================finished===================")
