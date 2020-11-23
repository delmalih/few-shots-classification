##########################
# Imports
##########################


# Built-in
import os
import shutil

# Global
import cv2
import numpy as np

# Local
from few_shots_clf.utils import get_all_images_from_folder
from few_shots_clf.test.test_utils import TEST_DIRECTORY_PATH
from few_shots_clf.test.test_utils import empty_dir


##########################
# Function
##########################


def test_empty_folder():
    """[summary]
    """
    # Empty dir
    empty_dir()

    # Get all images from folder
    paths = get_all_images_from_folder(TEST_DIRECTORY_PATH)

    # Assert
    assert len(paths) == 0


def test_folder_with_no_images():
    """[summary]
    """
    # Empty dir
    empty_dir()

    # Add a file
    file_path = os.path.join(TEST_DIRECTORY_PATH, "tmp.txt")
    with open(file_path, "w") as tmp_file:
        tmp_file.write("test")

    # Get all images from folder
    paths = get_all_images_from_folder(TEST_DIRECTORY_PATH)

    # Assert
    assert len(paths) == 0

    # Delete
    os.remove(file_path)
    shutil.rmtree(TEST_DIRECTORY_PATH)


def test_folder_with_one_image():
    """[summary]
    """
    # Empty dir
    empty_dir()

    # Create image
    img = np.zeros((10, 10, 3))

    # Add an image
    img_path = os.path.join(TEST_DIRECTORY_PATH, "tmp.jpg")
    cv2.imwrite(img_path, img)

    # Get all images from folder
    paths = get_all_images_from_folder(TEST_DIRECTORY_PATH)

    # Assert
    assert len(paths) == 1
    assert img_path in paths

    # Delete
    os.remove(img_path)
    shutil.rmtree(TEST_DIRECTORY_PATH)


def test_folder_with_multiple_images():
    """[summary]
    """
    # Empty dir
    empty_dir()

    # Number of images
    nb_images = 10

    # Loop image creation
    img_paths = []
    for k in range(nb_images):
        # Create image
        img = np.zeros((10, 10, 3))

        # Add image
        img_path = os.path.join(TEST_DIRECTORY_PATH, f"tmp{k}.jpg")
        cv2.imwrite(img_path, img)

        # Update img_paths list
        img_paths.append(img_path)

    # Get all images from folder
    paths = get_all_images_from_folder(TEST_DIRECTORY_PATH)

    # Assert
    assert len(paths) == nb_images
    for img_path in img_paths:
        assert img_path in paths

    # Delete
    for img_path in img_paths:
        os.remove(img_path)
    shutil.rmtree(TEST_DIRECTORY_PATH)
