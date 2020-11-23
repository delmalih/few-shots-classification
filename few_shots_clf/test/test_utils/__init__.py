##########################
# Imports
##########################


# Built-in
import os
import shutil

# Local
from few_shots_clf.test import TEST_DIRECTORY_PATH as DIR_PATH


##########################
# Function
##########################


TEST_DIRECTORY_PATH = os.path.join(DIR_PATH, "test_utils")


def empty_dir():
    """[summary]
    """
    # Empty dir
    if os.path.exists(TEST_DIRECTORY_PATH):
        shutil.rmtree(TEST_DIRECTORY_PATH)

    # Create dir
    if not os.path.exists(TEST_DIRECTORY_PATH):
        os.makedirs(TEST_DIRECTORY_PATH)
