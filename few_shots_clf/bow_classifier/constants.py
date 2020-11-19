# pylint: disable=no-member

##########################
# Imports
##########################


import os
import cv2


##########################
# Constants
##########################


VERBOSE = True  # boolean (True or False)

FEATURE_EXTRACTOR = cv2.xfeatures2d.SURF_create(extended=True)

IMAGE_SIZE = 256  # int

KEYPOINT_STRIDE = 8  # int

KEYPOINT_SIZES = [12, 24, 32, 48, 56, 64]  # List of ints

VOCAB_SIZE = 2000

TMP_FOLDER_PATH = "/tmp/few_shots_clf/bow_classifier/"  # existing path

VOCAB_PATH = os.path.join(TMP_FOLDER_PATH,
                          "vocab.pickle")

CATALOG_FEATURES_PATH = os.path.join(TMP_FOLDER_PATH,
                                     "catalog-features.pickle")

VOCAB_FINGERPRINT_PATH = os.path.join(TMP_FOLDER_PATH,
                                      "vocab_fingerprint.pickle")
