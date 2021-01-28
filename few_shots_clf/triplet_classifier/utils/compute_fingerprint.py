##########################
# Imports
##########################


from typing import Dict
from hashlib import sha224

from few_shots_clf.utils import compute_catalog_fingerprint


##########################
# Function
##########################


def compute_fingerprint(catalog_path: str, config: Dict) -> str:
    """Computes the fingerprint of a the FMClassifier given the catalog path and its config.

    Args:
        catalog_path (str): The path of the input catalog.
        config (Dict): The config of the classifier.

    Returns:
        str: The fingerprint of the classifier.
    """
    # Catalog fingerprint
    catalog_fingerprint = compute_catalog_fingerprint(catalog_path,
                                                      verbose=config.verbose)

    # Config fingerprint
    config_fingerprint = compute_config_fingerprint(config)

    # Final fingerprint
    fingerprint = f"{catalog_fingerprint}{config_fingerprint}"
    fingerprint = sha224(str.encode(fingerprint)).hexdigest()

    return fingerprint


def compute_config_fingerprint(config: Dict) -> str:
    """Computes the config fingerprint.

    Args:
        config (Dict): The config of the classifier.

    Returns:
        str: the config fingerprint.
    """
    # Init config fingerprint
    config_fingerprint = ""

    # Add image_size
    config_fingerprint = f"{config_fingerprint}{config.image_size}"

    # Add mining_strategy
    config_fingerprint = f"{config_fingerprint}{config.mining_strategy}"

    # Add embedding_size
    config_fingerprint = f"{config_fingerprint}{config.embedding_size}"

    return config_fingerprint
