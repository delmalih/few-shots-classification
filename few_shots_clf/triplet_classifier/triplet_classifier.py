# pylint: disable=attribute-defined-outside-init, no-member, line-too-long

##########################
# Imports
##########################


import os
from typing import Dict

import pickle
import numpy as np
from tensorflow import keras
from easydict import EasyDict as edict

from few_shots_clf import utils
from few_shots_clf.triplet_classifier import constants
from few_shots_clf.triplet_classifier import utils as triplet_utils


##########################
# TripletClassifier
##########################


class TripletClassifier:
    """Class implementing the Classifier trained on triplet loss (TripletClassifier)

    Args:
        catalog_path (string): [description]
        params (dict): [description]
    """

    ##########################
    # Init
    ##########################

    def __init__(self, catalog_path: str, params: Dict = {}):
        self.catalog_path = catalog_path
        self._config_classifier(catalog_path, params)

    ##########################
    # Config
    ##########################

    def _config_classifier(self, catalog_path, params):
        self._get_classifier_config(params)
        self._get_catalog_images(catalog_path)
        self._get_catalog_labels(catalog_path)
        self._get_catalog_images2labels()
        self._get_triplet_model()
        self._compile_triplet_model()
        self._load_fingerprints()

    def _get_classifier_config(self, params):
        self.config = edict({
            "verbose": params.get("verbose", constants.VERBOSE),
            "image_size": params.get("image_size", constants.IMAGE_SIZE),
            "triplet_margin": params.get("triplet_margin", constants.TRIPLET_MARGIN),
            "mining_strategy": params.get("mining_strategy", constants.MINING_STRATEGY),
            "embedding_size": params.get("embedding_size", constants.EMBEDDING_SIZE),
            "basic_batch_size": params.get("basic_batch_size", constants.BASIC_BATCH_SIZE),
            "augment_factor": params.get("augment_factor", constants.AUGMENT_FACTOR),
            "batch_size": params.get("batch_size", constants.BATCH_SIZE),
            "n_epochs": params.get("n_epochs", constants.N_EPOCHS),
            "model_backbone": params.get("model_backbone", constants.MODEL_BACKBONE),
            "learning_rate": params.get("learning_rate", constants.LEARNING_RATE),
            "model_path": params.get("model_path", constants.MODEL_PATH),
            "fingerprint_path": params.get("fingerprint_path", constants.FINGERPRINT_PATH),
        })

    def _get_catalog_images(self, catalog_path):
        self.catalog_images = utils.get_all_images_from_folder(catalog_path)
        if self.config.verbose:
            print(f"Found {len(self.catalog_images)} images!")

    def _get_catalog_labels(self, catalog_path):
        self.catalog_labels = utils.get_labels_from_catalog(catalog_path)
        if self.config.verbose:
            print(f"Found {len(self.catalog_labels)} labels!")

    def _get_catalog_images2labels(self):
        self.catalog_images2labels = utils.compute_images2labels(self.catalog_images,
                                                                 self.catalog_labels)

    def _get_triplet_model(self):
        self.triplet_model = triplet_utils.TripletModel(self.config.embedding_size,
                                                        self.config.model_backbone)
        self.triplet_model.build(input_shape=(self.config.batch_size,
                                              self.config.image_size,
                                              self.config.image_size,
                                              3))
        if self.config.verbose:
            self.triplet_model.summary()

    def _compile_triplet_model(self):
        triplet_loss = triplet_utils.triplet_loss_function(self.config.triplet_margin,
                                                           self.config.mining_strategy)
        triplet_metric = triplet_utils.triplet_loss_metric(
            self.config.triplet_margin)
        self.triplet_model.compile(optimizer=keras.optimizers.Adam(lr=self.config.learning_rate),
                                   loss=triplet_loss,
                                   metrics=[triplet_metric])

    def _load_fingerprints(self):
        # Previous fingerprint
        if os.path.exists(self.config.fingerprint_path):
            with open(self.config.fingerprint_path, "rb") as pickle_file:
                self.config.fingerprint = pickle.load(pickle_file)
        else:
            self.config.fingerprint = ""

        # Current fingerprint
        self.fingerprint = triplet_utils.compute_fingerprint(self.catalog_path,
                                                             self.config)

    ##########################
    # Train
    ##########################

    def train(self):
        """Method used to train the classifier.
        """
        train_generator = self._get_data_generator()
        reduce_lr_on_plateau_callback = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                          verbose=self.config.verbose)
        checkpointer_callback = keras.callbacks.ModelCheckpoint(self.config.model_path,
                                                                save_best_only=True,
                                                                monitor='loss',
                                                                verbose=self.config.verbose)
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                                patience=10,
                                                                verbose=self.config.verbose)
        self.triplet_model.fit_generator(generator=train_generator,
                                         epochs=self.config.n_epochs,
                                         verbose=self.config.verbose,
                                         use_multiprocessing=False,
                                         callbacks=[reduce_lr_on_plateau_callback,
                                                    checkpointer_callback,
                                                    early_stopping_callback])

    def _get_data_generator(self) -> triplet_utils.DataGenerator:
        catalog_labels = list(
            map(lambda img: self.catalog_images2labels[img], self.catalog_images))
        catalog_label_ids = np.float32(
            list(map(self.label_str2id, catalog_labels)))
        return triplet_utils.DataGenerator(self.catalog_images,
                                           catalog_label_ids,
                                           self.config.image_size,
                                           self.config.basic_batch_size,
                                           self.config.augment_factor)

    ##########################
    # Predict
    ##########################

    ##########################
    # Utils
    ##########################

    def label_id2str(self, label_id: int) -> str:
        """Gets the label_str given the label_id.

        Args:
            label_id (int): The given label_id.

        Returns:
            str: The label_str of the given label_id.
        """
        return self.catalog_labels[label_id]

    def label_str2id(self, label_str: str) -> int:
        """Gets the label_id given the label_str.

        Args:
            label_str (str): The given label_str.

        Returns:
            int: The label_id of the given label_id.
        """
        if label_str in self.catalog_labels:
            return self.catalog_labels.index(label_str)
        return -1
