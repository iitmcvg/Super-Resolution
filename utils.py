"""
Some codes from https://github.com/Newmu/dcgan_code and https://github.com/Tetrachrome/subpixel
"""
from __future__ import division
import math
import random
import pprint
import scipy.misc
from scipy.misc import imresize
import numpy as np
import tensorflow as tf

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)

def save_images(images, size, image_path):
    num_im = size[0] * size[1]
    return imsave(inverse_transform(images[:num_im]), size, image_path)

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    # print('IMAGE SHAPE: ', images.shape)
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        # print('IMAGESHAPPPPPPPE ', images.shape)
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w=None, resize_w=128):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=128, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (np.array(images)+1.)/2.

#For dividing true input into equal grids of 32*32
def make_grid_copied(arr, nrows, ncols):
    """
    Faster alternative to make_grid that doesn't work now :P
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    
    channels=list()
    for ch in range(0,arr.shape[2]):
        channels.append(arr[:,:,ch])
    
    final=list()
    h, w= arr.shape[:2]
    for chn in channels:
        final.append(chn.reshape(h//nrows, nrows, w//ncols, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))
    
    final=np.array(final).swapaxes(0,1).swapaxes(1,2).swapaxes(2,3)
    print('Shape of reshaped grids',final.shape)
    return final
    
def make_grid(img):
    #Breaking into 32*32 images.
    grids=list()
    h,w = img.shape[:2]
    print('Image shape: ',h,w)
    #nrows = h//32
    #ncols = w//32

    padh=32-int(h%32)
    print('Row padding: ',padh)
    padw=32-int(w%32)
    print('Column padding: ',padw)
    neo_image=np.zeros((h+padh,w+padw,3))
    neo_image[padh//2:h+padh//2,padw//2:w+padw//2,:]=img
    print('Neo Image size: ',neo_image.shape[:2])
    nrows=neo_image.shape[0]//32
    ncols=neo_image.shape[1]//32
    
    #num_grid=image.shape[0]*image.shape[1]/(32*32)
    for i in range(0,nrows):
        for j in range(0,ncols):
            grids.append(neo_image[32*i:32*(i+1),32*j:32*(j+1),:])
    
    print('Shape of single grid: ',grids[0].shape)
    return grids,nrows,ncols

def join_grid(output_list,nrows,ncols):
    # h,w=output_list[0].shape[:2]
    large_ass_output=np.zeros((128*nrows,128*ncols,3))
    for i in range(0,nrows):
        for j in range(0,ncols):
            large_ass_output[128*i:128*(i+1),128*j:128*(j+1),:]=output_list[i*ncols+j]

    return large_ass_output

def doresize(x, shape):
    x = np.copy((x+1.)*127.5).astype("uint8")
    y = imresize(x, shape)
    return y

def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def phase_shift_deconv(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    else:
        X = _phase_shift(X, r)
    return X
