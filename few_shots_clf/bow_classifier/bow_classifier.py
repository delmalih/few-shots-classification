# pylint: disable=attribute-defined-outside-init, no-member

##########################
# Imports
##########################


# Global
import os
import pickle
import numpy as np
from easydict import EasyDict as edict

# few_shots_clf
from few_shots_clf import utils
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

    # def _get_catalog_images(self, catalog_path):
    #     self.catalog_images = utils.get_all_images_from_folder(catalog_path)

    # def _get_catalog_labels(self, catalog_path):
    #     self.catalog_labels = utils.get_labels_from_catalog(catalog_path)

    # def _get_catalog_images2labels(self):
    #     self.catalog_images2labels = utils.compute_images2labels(self.catalog_images,
    #                                                              self.catalog_labels)

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
            self._load_matcher_index()

    def _should_create_vocab(self):
        fingerprint_changed = self.config.fingerprint != self.fingerprint
        vocab_file_exists = os.path.isfile(self.config.matcher_path)
        return fingerprint_changed or (not vocab_file_exists)

    def _create_vocab(self):
        pass
        # # Get descriptors
        # catalog_descriptors = self._get_catalog_descriptors()

        # # Config matcher
        # if self.config.verbose:
        #     print("Creating Index ...")
        # self.matcher.addDataPointBatch(catalog_descriptors)
        # self.matcher.createIndex(self.config.matcher_index_params,
        #                          print_progress=self.config.verbose)
        # self.matcher.setQueryTimeParams(self.config.matcher_query_params)

    def _get_catalog_descriptors(self):
        # Init descriptors list
        catalog_descriptors = []

        # Init iterator
        iterator = utils.get_iterator(
            utils.get_all_images_from_folder(self.catalog_path),
            verbose=self.config.verbose,
            description="Computing catalog descriptors")

        # Compute all descriptors
        for path in iterator:
            # Read image
            img = utils.read_image(path, size=self.config.image_size)

            # Compute keypoints
            keypoints = utils.compute_keypoints(
                img,
                self.config.keypoint_stride,
                self.config.keypoint_sizes)

            # Compute descriptors
            descriptors = utils.compute_descriptors(
                img,
                keypoints,
                self.config.feature_extractor)

            # Update descriptors list
            catalog_descriptors.append(descriptors)

        # Reshape descriptors list
        catalog_descriptors = np.array(catalog_descriptors)
        catalog_descriptors = catalog_descriptors.reshape(-1,
                                                          catalog_descriptors.shape[-1])

        return catalog_descriptors

    def _save_matcher_index(self):
        matcher_folder = "/".join(self.config.matcher_path.split("/")[:-1])
        if not os.path.exists(matcher_folder):
            os.makedirs(matcher_folder)
        self.matcher.saveIndex(self.config.matcher_path)

    def _save_fingerprint(self):
        fingerprint_folder = "/".join(
            self.config.fingerprint_path.split("/")[:-1])
        if not os.path.exists(fingerprint_folder):
            os.makedirs(fingerprint_folder)
        with open(self.config.fingerprint_path, "wb") as pickle_file:
            pickle.dump(self.fingerprint, pickle_file)

    def _load_matcher_index(self):
        if self.config.verbose:
            print("Loading Index...")
        self.matcher.loadIndex(self.config.matcher_path)

    ##########################
    # Predict
    ##########################

    def predict(self, query_path):
        """[summary]

        Args:
            query_path ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Read img
        query_img = utils.read_image(query_path, size=self.config.image_size)

        # Get keypoints
        query_keypoints = utils.compute_keypoints(query_img,
                                                  self.config.keypoint_stride,
                                                  self.config.keypoint_sizes)

        # Get descriptors
        query_descriptors = utils.compute_descriptors(query_img,
                                                      query_keypoints,
                                                      self.config.feature_extractor)

        # Get scores
        scores = self._get_query_scores(query_descriptors)

        # To numpy
        scores = np.array(scores)

        return scores

    def predict_batch(self, query_paths):
        """[summary]

        Args:
            query_paths ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Init scores
        scores = []

        # Get iterator
        iterator = utils.get_iterator(query_paths,
                                      verbose=self.config.verbose,
                                      description="Prediction of all queries")

        # Loop over all queries
        for query_path in iterator:
            # Predict score of query
            query_scores = self.predict(query_path)

            # Update scores
            scores.append(query_scores)

        # To numpy
        scores = np.array(scores)

        return scores

    def _get_query_scores(self, query_descriptors):
        # Init scores variables
        scores = np.zeros((len(self.catalog_labels)))
        n_desc = query_descriptors.shape[0]

        # Compute matches
        train_idx, distances = self._compute_query_matches(query_descriptors)

        # Compute score matrix
        scores_matrix = self._compute_scores_matrix(distances)

        # Compute final scores
        for ind, nn_train_idx in enumerate(train_idx):
            for k, idx in enumerate(nn_train_idx):
                # Get image_path
                image_path = self.catalog_images[idx // n_desc]

                # Get image_label
                image_label = self.catalog_images2labels[image_path]

                # Get label_idx
                label_idx = self.catalog_labels.index(image_label)

                # Update score
                scores[label_idx] += scores_matrix[ind, k]

        return scores

    def _compute_query_matches(self, query_descriptors):
        # Compute matches
        matches = self.matcher.knnQueryBatch(
            query_descriptors,
            k=self.config.k_nn)

        # Separate matches into train_idx and distances
        train_idx = np.array([m[0] for m in matches])
        distances = np.array([m[1] for m in matches])

        return train_idx, distances

    def _compute_scores_matrix(self, distances):
        if self.config.scoring == "distance":
            return self._compute_scores_matrix_distance(distances)
        elif self.config.scoring == "count":
            return self._compute_scores_matrix_count(distances)
        else:
            return self._compute_scores_matrix_distance(distances)

    def _compute_scores_matrix_distance(self, distances):
        return np.exp(-distances)

    def _compute_scores_matrix_count(self, distances):
        scores_matrix = np.zeros(distances.shape)
        for k in range(self.config.k_nn):
            scores_matrix[:, k] = self.config.k_nn - k
        return scores_matrix
