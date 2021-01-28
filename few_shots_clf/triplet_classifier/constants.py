# pylint: disable=no-member

##########################
# Imports
##########################


import os


##########################
# Constants
##########################


VERBOSE = True  # boolean (True or False)

IMAGE_SIZE = 256  # int

TRIPLET_MARGIN = 1.0  # float

MINING_STRATEGY = "soft"  # str ("soft" or "hard")

EMBEDDING_SIZE = 128

BATCH_SIZE = 1

N_EPOCHS = 10

TMP_FOLDER_PATH = "/tmp/few_shots_clf/triplet_classifier/"  # existing path

FINGERPRINT_PATH = os.path.join(TMP_FOLDER_PATH,
                                "fingerprint.pickle")
