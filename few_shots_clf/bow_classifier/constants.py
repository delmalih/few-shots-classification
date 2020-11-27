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

FEATURE_DESCRIPTOR = cv2.xfeatures2d.SURF_create(extended=True)

# FEATURE_DIMENSION = 128 * 3

VOCAB_SIZE = 2000  # int

IMAGE_SIZE = 256  # int

KEYPOINT_STRIDE = 8  # int

KEYPOINT_SIZES = [12, 24, 32, 48, 56, 64]  # List of ints

TMP_FOLDER_PATH = "/tmp/few_shots_clf/bow_classifier/"  # existing path

FINGERPRINT_PATH = os.path.join(TMP_FOLDER_PATH,
                                "fingerprint.pickle")

VOCAB_PATH = os.path.join(TMP_FOLDER_PATH,
                          "vocab.pickle")

CATALOG_FEATURES_PATH = os.path.join(TMP_FOLDER_PATH,
                                     "catalog_features_path.pickle")

# MATCHER_DISTANCE = "angular"

# MATCHER_N_TREES = 10

# MATCHER_PATH = os.path.join(TMP_FOLDER_PATH,
#                             "matcher-classifier-custom.ann")


# SCORING = "count"  # Can be "distance" or "count"

# K_NN = 1
