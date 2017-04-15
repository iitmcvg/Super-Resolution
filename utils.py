"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

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
    #nrows = h//32
    #ncols = w//32

    padh=int(h%32)
    print('Row padding: ',padh)
    padw=int(w%32)
    print('Column padding: ',padw)
    neo_image=np.zeros((h+padh,w+padw,3))
    neo_image[padh//2:h+padh//2,padw//2:w+padw//2,:]=img
    
    nrows=neo_image.shape[0]//32
    ncols=neo_image.shape[1]//32
    
    #num_grid=image.shape[0]*image.shape[1]/(32*32)
    for i in range(0,ncols):
        for j in range(0,nrows):
            grids.append(neo_image[32*i:32*(i+1),32*j:32*(j+1),:])
    
    print('Shape of single grid: ',grids[0].shape)
    return grids,nrows,ncols

def join_grid(output_list,nrows,ncols):
    # h,w=output_list[0].shape[:2]
    large_ass_output=np.zeros((128*nrows,128*ncols,3))
    for i in range(0,ncols):
        for j in range(0,nrows):
            large_ass_output[128*i:128*(i+1),128*j:128*(j+1),:]=output_list[i*ncols+j]

    return large_ass_output
