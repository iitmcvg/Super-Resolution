from __future__ import print_function
import os
import sys
import gzip
import json
import shutil
import zipfile
import argparse
import subprocess
from six.moves import urllib

def download_celeb_a(dirpath):
    NUM_EXAMPLES = 202599
    TRAIN_STOP = 162770
    VALID_STOP = 182637
    data_dir = 'celebA'
    # if os.path.exists(os.path.join(dirpath, data_dir)):
    #     print('Found Celeb-A - skip')
    #     return
    
    '''
    url = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1&pv=1'
    filepath = download(url, dirpath)
    zip_dir = ''
    with zipfile.ZipFile(filepath) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(filepath)
    '''
    zip_dir='celebA'
    # now split data into train/valid/test
    train_dir = os.path.join(dirpath, 'train')
    valid_dir = os.path.join(dirpath, 'valid')
    test_dir = os.path.join(dirpath, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    zip_path = os.path.join(dirpath, zip_dir)
    for i in range(NUM_EXAMPLES):
        image_filename = "{:06d}.jpg".format(i+1)
        candidate_file = os.path.join(zip_path, image_filename)
        print(candidate_file)
        if os.path.exists(candidate_file):
            if i < TRAIN_STOP:
                dest_dir = train_dir
            elif i < VALID_STOP:
                dest_dir = valid_dir
            else:
                dest_dir = test_dir
            dest_file = os.path.join(dest_dir, image_filename)
            os.rename(candidate_file, dest_file)

    # os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))

download_celeb_a("data/")
