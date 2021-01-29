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

BASIC_BATCH_SIZE = 10

AUGMENT_FACTOR = 5

BATCH_SIZE = BASIC_BATCH_SIZE * AUGMENT_FACTOR

N_EPOCHS = 200

MODEL_BACKBONE = "vgg16"

LEARNING_RATE = 1e-4

TMP_FOLDER_PATH = "/tmp/few_shots_clf/triplet_classifier/"  # existing path

FINGERPRINT_PATH = os.path.join(TMP_FOLDER_PATH,
                                "fingerprint.pickle")
