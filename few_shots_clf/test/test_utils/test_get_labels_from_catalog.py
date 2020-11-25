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
from few_shots_clf.utils import get_labels_from_catalog
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

    # Get catalog_path
    catalog_path = os.path.join(TEST_DIRECTORY_PATH, "catalog")
    os.makedirs(catalog_path)

    # Get label_path
    label_name = "label"
    label_path = os.path.join(catalog_path, label_name)
    os.makedirs(label_path)

    # Get all labels from catalog
    labels = get_labels_from_catalog(catalog_path)

    # Assert
    assert len(labels) == 0

    # Delete
    shutil.rmtree(label_path)
    shutil.rmtree(catalog_path)
    shutil.rmtree(TEST_DIRECTORY_PATH)


def test_folder_with_no_images():
    """[summary]
    """
    # Empty dir
    empty_dir()

    # Get catalog_path
    catalog_path = os.path.join(TEST_DIRECTORY_PATH, "catalog")
    os.makedirs(catalog_path)

    # Get label_path
    label_name = "label"
    label_path = os.path.join(catalog_path, label_name)
    os.makedirs(label_path)

    # Add a file
    file_path = os.path.join(label_path, "tmp.txt")
    with open(file_path, "w") as tmp_file:
        tmp_file.write("test")

    # Get all labels from catalog
    labels = get_labels_from_catalog(catalog_path)

    # Assert
    assert len(labels) == 0

    # Delete
    os.remove(file_path)
    shutil.rmtree(label_path)
    shutil.rmtree(catalog_path)
    shutil.rmtree(TEST_DIRECTORY_PATH)


def test_folder_with_one_label():
    """[summary]
    """
    # Empty dir
    empty_dir()

    # Get catalog_path
    catalog_path = os.path.join(TEST_DIRECTORY_PATH, "catalog")
    os.makedirs(catalog_path)

    # Get label_path
    label_name = "label"
    label_path = os.path.join(catalog_path, label_name)
    os.makedirs(label_path)

    # Create image
    img = np.zeros((10, 10, 3))

    # Add an image
    img_path = os.path.join(label_path, "tmp.jpg")
    cv2.imwrite(img_path, img)

    # Get all labels from catalog
    labels = get_labels_from_catalog(catalog_path)

    # Assert
    assert len(labels) == 1
    assert label_name in labels

    # Delete
    os.remove(img_path)
    shutil.rmtree(label_path)
    shutil.rmtree(catalog_path)
    shutil.rmtree(TEST_DIRECTORY_PATH)


def test_folder_with_multiple_labels():
    """[summary]
    """
    # Empty dir
    empty_dir()

    # Get catalog_path
    catalog_path = os.path.join(TEST_DIRECTORY_PATH, "catalog")
    os.makedirs(catalog_path)

    # Number of labels
    nb_labels = 10

    # Get labels_paths
    label_names = [f"label{k}"
                   for k in range(nb_labels)]
    label_paths = [os.path.join(catalog_path, label_name)
                   for label_name in label_names]
    for label_path in label_paths:
        os.makedirs(label_path)

    # Loop over label_paths
    img_paths = []
    for label_path in label_paths:
        for k in range(np.random.randint(1, 5)):
            # Create image
            img = np.zeros((10, 10, 3))

            # Save image
            img_path = os.path.join(label_path, f"tmp{k}.jpg")
            cv2.imwrite(img_path, img)

            # Update img_paths list
            img_paths.append(img_path)

    # Get all labels from catalog
    labels = get_labels_from_catalog(catalog_path)

    # Assert
    assert len(labels) == nb_labels
    for label_name in label_names:
        assert label_name in labels

    # Delete
    for img_path in img_paths:
        os.remove(img_path)
    for label_path in label_paths:
        shutil.rmtree(label_path)
    shutil.rmtree(catalog_path)
    shutil.rmtree(TEST_DIRECTORY_PATH)
