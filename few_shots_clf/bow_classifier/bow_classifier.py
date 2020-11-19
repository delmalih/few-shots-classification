# pylint: disable=attribute-defined-outside-init, no-member

##########################
# Imports
##########################


# Global
import os
import pickle
from easydict import EasyDict as edict

# few_shots_clf
from few_shots_clf.bow_classifier import utils as bow_utils
from few_shots_clf.bow_classifier import constants


##########################
# bow_classifier
##########################


class BOWClassifier:
    """[summary]

    Args:
        catalog_path (string): [description]
        params (dict): [description]
    """

    ##########################
    # Init
    ##########################

    def __init__(self, catalog_path, params={}):
        self.catalog_path = catalog_path
        self._config_classifier(params)

    ##########################
    # Config
    ##########################

    def _config_classifier(self, params):
        self._get_classifier_config(params)
        # self._get_catalog_images(catalog_path)
        # self._get_catalog_labels(catalog_path)
        # self._get_catalog_images2labels()
        self._load_fingerprints()

    def _get_classifier_config(self, params):
        self.config = edict({
            "verbose": params.get("verbose", constants.VERBOSE),
            "feature_extractor": params.get("feature_extractor", constants.FEATURE_EXTRACTOR),
            "image_size": params.get("image_size", constants.IMAGE_SIZE),
            "keypoint_stride": params.get("keypoint_stride", constants.KEYPOINT_STRIDE),
            "keypoint_sizes": params.get("keypoint_sizes", constants.KEYPOINT_SIZES),
            "vocab_size": params.get("vocab_size", constants.VOCAB_SIZE),
            "vocab_path": params.get("vocab_path", constants.VOCAB_PATH),
            "catalog_features_path": params.get("catalog_features_path",
                                                constants.CATALOG_FEATURES_PATH),
            "vocab_fingerprint_path": params.get("vocab_fingerprint_path",
                                                 constants.VOCAB_FINGERPRINT_PATH)
        })

    def _load_fingerprints(self):
        # Previous fingerprint
        if os.path.exists(self.config.fingerprint_path):
            with open(self.config.fingerprint_path, "rb") as pickle_file:
                self.config.fingerprint = pickle.load(pickle_file)
        else:
            self.config.fingerprint = ""

        # Current fingerprint
        self.fingerprint = bow_utils.compute_fingerprint(self.catalog_path,
                                                         self.config)

    ##########################
    # Train
    ##########################

    def train(self):
        """[summary]
        """
        # Create or load vocab
        if self._should_create_vocab():
            self._create_vocab()
            self._save_vocab()
            self._save_fingerprint()
        else:
            self._load_vocab()

    def _should_create_vocab(self):
        fingerprint_changed = self.config.fingerprint != self.fingerprint
        vocab_file_exists = os.path.isfile(self.config.matcher_path)
        return fingerprint_changed or (not vocab_file_exists)

    def _create_vocab(self):
        pass

    def _save_vocab(self):
        pass

    def _save_fingerprint(self):
        fingerprint_folder = "/".join(
            self.config.fingerprint_path.split("/")[:-1])
        if not os.path.exists(fingerprint_folder):
            os.makedirs(fingerprint_folder)
        with open(self.config.fingerprint_path, "wb") as pickle_file:
            pickle.dump(self.fingerprint, pickle_file)

    def _load_vocab(self):
        pass
