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


TEST_DIRECTORY_PATH = "/tmp/test_few_shots_clf"


def test_empty_folder():
    """[summary]
    """
    # Directory path
    dir_path = f"{TEST_DIRECTORY_PATH}/test_empty_folder/"

    # Create empty dir
    os.makedirs(dir_path)

    # Get all images from folder
    paths = get_all_images_from_folder(dir_path)

    # Assert
    assert len(paths) == 0

    # Delete directory
    shutil.rmtree(dir_path)
    shutil.rmtree(TEST_DIRECTORY_PATH)


def test_folder_with_no_images():
    """[summary]
    """
    # Directory path
    dir_path = f"{TEST_DIRECTORY_PATH}/test_folder_with_no_images/"

    # Create empty dir
    os.makedirs(dir_path)

    # Add a file
    with open(f"{dir_path}/tmp.txt", "w") as tmp_file:
        tmp_file.write("test")

    # Get all images from folder
    paths = get_all_images_from_folder(dir_path)

    # Assert
    assert len(paths) == 0

    # Delete directory
    shutil.rmtree(dir_path)
    shutil.rmtree(TEST_DIRECTORY_PATH)
