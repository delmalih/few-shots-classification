##########################
# Imports
##########################


# Built-in
import os
import shutil

# Global
from few_shots_clf.utils import get_all_images_from_folder


##########################
# Function
##########################


def test_empty_folder():
    """[summary]
    """
    # Directory path
    dir_path = "/tmp/test_few_shots_clf/test_empty_folder/"

    # Create empty dir
    os.makedirs(dir_path)

    # Get all images from folder
    paths = get_all_images_from_folder(dir_path)

    # Assert
    assert len(paths) == 0

    # Delete directory
    shutil.rmtree(dir_path)
