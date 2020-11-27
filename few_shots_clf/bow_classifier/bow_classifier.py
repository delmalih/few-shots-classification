# pylint: disable=attribute-defined-outside-init, no-member, no-self-use, too-many-instance-attributes

##########################
# Imports
##########################


# Global
import os
import pickle
import numpy as np
from annoy import AnnoyIndex
from easydict import EasyDict as edict
from sklearn.cluster import MiniBatchKMeans

# few_shots_clf
from few_shots_clf import utils
from few_shots_clf.fm_classifier import constants
from few_shots_clf.fm_classifier import utils as bow_utils


##########################
# BOWClassifier
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
        self._config_classifier(catalog_path, params)

    ##########################
    # Config
    ##########################

    def _config_classifier(self, catalog_path, params):
        self._get_classifier_config(params)
        self._get_catalog_images(catalog_path)
        self._get_catalog_labels(catalog_path)
        self._get_catalog_images2labels()
        self._load_fingerprints()

    def _get_classifier_config(self, params):
        self.config = edict({
            "verbose": params.get("verbose", constants.VERBOSE),
            "feature_descriptor": params.get("feature_descriptor", constants.FEATURE_DESCRIPTOR),
            "vocab_size": params.get("vocab_size", constants.VOCAB_SIZE),
            "image_size": params.get("image_size", constants.IMAGE_SIZE),
            "keypoint_stride": params.get("keypoint_stride", constants.KEYPOINT_STRIDE),
            "keypoint_sizes": params.get("keypoint_sizes", constants.KEYPOINT_SIZES),
            "vocab_path": params.get("vocab_path",
                                     constants.VOCAB_PATH),
            "fingerprint_path": params.get("fingerprint_path",
                                           constants.FINGERPRINT_PATH),
        })

    def _get_catalog_images(self, catalog_path):
        self.catalog_images = utils.get_all_images_from_folder(catalog_path)

    def _get_catalog_labels(self, catalog_path):
        self.catalog_labels = utils.get_labels_from_catalog(catalog_path)

    def _get_catalog_images2labels(self):
        self.catalog_images2labels = utils.compute_images2labels(self.catalog_images,
                                                                 self.catalog_labels)

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
        # Init matcher
        self.matcher = AnnoyIndex(self.config.feature_dimension,
                                  self.config.matcher_distance)

        # Create or load matcher
        if self._should_create_vocab():
            self._create_vocab()
            self._save_vocab()
            self._save_fingerprint()
        else:
            self._load_vocab()

    def _should_create_vocab(self):
        fingerprint_changed = self.config.fingerprint != self.fingerprint
        vocab_file_exists = os.path.isfile(self.config.vocab_path)
        return fingerprint_changed or (not vocab_file_exists)

    def _create_vocab(self):
        # Init vocab
        self.vocab = {}

        # Get descriptors
        catalog_descriptors, catalog_descriptors_labels = self._get_catalog_descriptors()

        # KMeans
        if self.config.verbose:
            print("KMeans step ...")
        kmeans = MiniBatchKMeans(n_clusters=self.config.vocab_size,
                                 init_size=3 * self.config.vocab_size)
        clusters = kmeans.fit_predict(catalog_descriptors)

        # Save features
        self.vocab["features"] = kmeans.cluster_centers_

        # Build vocab
        self._build_vocab_idf(clusters, catalog_descriptors_labels)

    def _get_catalog_descriptors(self):
        # Init descriptors list
        catalog_descriptors = []
        catalog_descriptors_labels = []

        # Init iterator
        iterator = utils.get_iterator(
            self.catalog_images,
            verbose=self.config.verbose,
            description="Computing catalog descriptors")

        # Compute all descriptors
        for k, path in enumerate(iterator):
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
                self.config.feature_descriptor)
            descriptors_labels = [k for _ in range(len(descriptors))]

            # Update descriptors list
            catalog_descriptors.append(descriptors)
            catalog_descriptors_labels.append(descriptors_labels)

        # Reshape descriptors list
        catalog_descriptors = np.array(catalog_descriptors)
        catalog_descriptors_labels = np.array(catalog_descriptors_labels)
        catalog_descriptors = catalog_descriptors.reshape(
            -1, catalog_descriptors.shape[-1])
        catalog_descriptors_labels = catalog_descriptors_labels.reshape(
            -1, catalog_descriptors_labels.shape[-1])

        return catalog_descriptors, catalog_descriptors_labels

    def _build_vocab_idf(self, clusters, descriptors_labels):
        # Compute IDF
        self.vocab["idf"] = np.zeros((self.vocab["features"].shape[0],))
        for cluster in set(clusters):
            # Get images of cluster
            catalog_images_in_cluster = set(
                descriptors_labels[clusters == cluster])

            # Compute cluster IDF
            idf_num = len(self.catalog_images_paths)
            idf_den = len(len(catalog_images_in_cluster))
            self.vocab["idf"][cluster] = np.log(idf_num / idf_den)

    def _save_vocab(self):
        vocab_folder = "/".join(self.config.vocab_path.split("/")[:-1])
        if not os.path.exists(vocab_folder):
            os.makedirs(vocab_folder)
        if self.config.verbose:
            print("Saving Vocab...")
        with open(self.config.vocab_path, "wb") as pickle_file:
            pickle.dump(self.vocab, pickle_file)

    def _load_vocab(self):
        if self.config.verbose:
            print("Loading Vocab...")
        with open(self.config.vocab_path, "rb") as pickle_file:
            self.vocab = pickle.load(pickle_file)
        self.matcher.load(self.config.matcher_path)

    def _save_fingerprint(self):
        fingerprint_folder = "/".join(
            self.config.fingerprint_path.split("/")[:-1])
        if not os.path.exists(fingerprint_folder):
            os.makedirs(fingerprint_folder)
        with open(self.config.fingerprint_path, "wb") as pickle_file:
            pickle.dump(self.fingerprint, pickle_file)

    ##########################
    # Predict
    ##########################

    def predict(self, query_path):
        """[summary]

        Args:
            query_path ([type]): [description]
        """
        return query_path

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

    ##########################
    # Metrics & Scores
    ##########################

    def score(self, query_paths, gt_labels):
        """[summary]

        Args:
            query_paths ([type]): [description]
            gt_labels ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Predict labels
        pred_scores = self.predict_batch(query_paths)
        pred_labels = np.argmax(pred_scores, axis=-1)

        # Accuracy
        gt_labels = np.array(gt_labels)
        nb_correct = (pred_labels == gt_labels).sum()
        nb_total = len(pred_labels)
        accuracy = nb_correct / nb_total

        return accuracy

    ##########################
    # Utils
    ##########################

    def label_id2str(self, label_id):
        """[summary]

        Args:
            label_id ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self.catalog_labels[label_id]

    def label_str2id(self, label_str):
        """[summary]

        Args:
            label_str ([type]): [description]

        Returns:
            [type]: [description]
        """
        if label_str in self.catalog_labels:
            return self.catalog_labels.index(label_str)
        return -1
