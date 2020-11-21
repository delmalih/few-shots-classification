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
from few_shots_clf.utils import read_image
from few_shots_clf.test.test_utils import TEST_DIRECTORY_PATH


##########################
# Function
##########################


def test_fixed_size():
    """[summary]
    """
    # Create dir
    if not os.path.exists(TEST_DIRECTORY_PATH):
        os.makedirs(TEST_DIRECTORY_PATH)

    # Create image
    img_h = np.random.randint(10, 100)
    img_w = np.random.randint(10, 100)
    img = np.zeros((img_h, img_w, 3))

    # Save image
    img_path = os.path.join(TEST_DIRECTORY_PATH, "tmp.jpg")
    cv2.imwrite(img_path, img)

    # Resize img
    resized_img = read_image(img_path, size=50)
    resized_h, resized_w, _ = resized_img.shape

    # Assert
    assert resized_w == 50
    assert resized_h == 50

    # Delete
    os.remove(img_path)
    shutil.rmtree(TEST_DIRECTORY_PATH)
