##########################
# Imports
##########################


# Global
import os
import cv2
import imutils
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

# Local
from utils import constants


##########################
# Functions
##########################


def get_all_files(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if ".DS_Store" not in file:
                file_path = os.path.join(r, file)
                files.append(file_path)
    return files


def get_iterator(iterator, verbose, desc=None):
    if verbose:
        iterator = tqdm(iterator, desc=desc)
    return iterator


def read_image(path, width=None, size=None):
    img = cv2.imread(path)
    if width is not None:
        img = imutils.resize(img, width=width)
    elif size is not None:
        img = cv2.resize(img, (size, size))
    return img


def get_keypoints(img, kpt_stride, kpt_sizes):
    return [
        cv2.KeyPoint(x, y, size)
        for size in kpt_sizes
        for x in range(0, img.shape[1], kpt_stride)
        for y in range(0, img.shape[0], kpt_stride)
    ]


def get_descriptors(img, keypoints, feature_extractor):
    _, descriptors = feature_extractor.compute(img, keypoints)
    return descriptors
