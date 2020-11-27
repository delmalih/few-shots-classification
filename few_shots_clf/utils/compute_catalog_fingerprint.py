##########################
# Imports
##########################


# Built-in
from hashlib import sha224

# Global
import cv2
import numpy as np

# Local
from few_shots_clf.utils import get_iterator
from few_shots_clf.utils import get_all_images_from_folder


##########################
# Function
##########################


def compute_catalog_fingerprint(catalog_path, verbose=True):
    """[summary]

    Args:
        catalog_path ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    # Init. catalog fingerprint
    catalog_fingerprint = ""

    # Get all paths
    image_paths = get_all_images_from_folder(catalog_path)

    # Iterator
    iterator = get_iterator(
        image_paths,
        verbose=verbose,
        description="Computing fingerprint...")

    # Loop over image_paths
    for image_path in iterator:
        # Hash image_path
        path_hash = sha224(str.encode(image_path)).hexdigest()

        # Read image
        img = cv2.imread(image_path)

        # Convert image to string
        img_str = np.array2string(img)

        # Hash image
        img_hash = sha224(str.encode(img_str)).hexdigest()

        # Compute image_fingerprint
        image_fingerprint = f"{path_hash}{img_hash}"

        # Update catalog_fingerprint
        catalog_fingerprint = f"{catalog_fingerprint}{image_fingerprint}"

    # Compute final fingerprint
    catalog_fingerprint = sha224(str.encode(catalog_fingerprint)).hexdigest()

    return catalog_fingerprint
