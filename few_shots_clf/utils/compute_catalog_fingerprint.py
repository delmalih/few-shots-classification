##########################
# Imports
##########################


from hashlib import sha224

import cv2
import numpy as np

from few_shots_clf.utils import get_all_images_from_folder
from few_shots_clf.utils import get_iterator


##########################
# Function
##########################


def compute_catalog_fingerprint(catalog_path: str, verbose=True) -> str:
    """Computes the fingerprint of a given catalog

    Args:
        catalog_path (str): The path of the input catalog.
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        str: the catalog fingerprint.
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
