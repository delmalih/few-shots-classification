##########################
# Imports
##########################


# Global
import os
import cv2
import nmslib
import numpy as np
from tqdm import tqdm

# Local (utils)
import utils

# Local (BaslineClassifier)
from Classifiers.BaselineClassifier import BaselineClassifier


##########################
# Classifier
##########################


class CustomClassifier(BaselineClassifier):
    def __init__(self, catalog_folder, params={}):
        super(CustomClassifier, self).__init__(
            catalog_folder,
            params=params)
        self.catalog_images_paths = utils.get_all_files(catalog_folder)
        self.config_matcher()

    def load_matcher(self):
        # Print
        if self.verbose:
            print("Loading index...")

        # Load index
        self.matcher.loadIndex(self.matcher_path_custom)

        # Print
        if self.verbose:
            print("Index loaded !")

    def create_matcher(self):
        # Get descriptors
        self.get_catalog_descriptors()

        # Print
        if self.verbose:
            print("Creating index...")

        # Config matcher
        self.matcher.addDataPointBatch(self.catalog_descriptors)
        self.matcher.createIndex(self.matcher_index_params,
                                 print_progress=self.verbose)
        self.matcher.setQueryTimeParams(self.matcher_query_params)

        # Print
        if self.verbose:
            print("Index created !")
            print("Saving index...")

        # Save Index
        self.matcher.saveIndex(self.matcher_path_custom)

        # Print
        if self.verbose:
            print("Index saved !")

    def config_matcher(self):
        # Init. matcher
        self.matcher = nmslib.init(method="hnsw", space="l2")

        # Create index
        if not self.force_matcher_compute and os.path.exists(self.matcher_path_custom):
            self.load_matcher()
        else:
            self.create_matcher()

    def get_catalog_descriptors(self):
        # Init. descriptors list
        self.catalog_descriptors = []

        # Init. iterator
        iterator = utils.get_iterator(
            self.catalog_images_paths,
            self.verbose,
            desc="Catalog description")

        # Compute descriptors
        for path in iterator:
            # Read image
            img = utils.read_image(path, size=self.image_size)

            # Compute keypoints
            keypoints = utils.get_keypoints(
                img,
                self.keypoint_stride,
                self.keypoint_sizes)

            # Compute descriptors
            descriptors = utils.get_descriptors(
                img,
                keypoints,
                self.feature_extractor)

            # Update descriptors list
            self.catalog_descriptors.append(descriptors)

        # Reshape descriptors list
        self.catalog_descriptors = np.array(self.catalog_descriptors)
        self.catalog_descriptors = self.catalog_descriptors.reshape(
            -1, self.catalog_descriptors.shape[-1])

    def get_query_scores(self, query_descriptors):
        # Compute matches
        matches = self.matcher.knnQueryBatch(
            query_descriptors,
            k=self.k_nn_custom)
        trainIdx = np.array([m[0] for m in matches])
        distances = np.array([m[1] for m in matches])

        # Compute scores
        scores = {}
        N_desc = query_descriptors.shape[0]
        scores_matrix = np.exp(-((distances / self.score_sigma) ** 2))
        for ind, nn_trainIdx in enumerate(trainIdx):
            for k, idx in enumerate(nn_trainIdx):
                catalog_path = self.catalog_images_paths[idx // N_desc]
                catalog_label = catalog_path.split("/")[-2]
                scores[catalog_label] = scores.get(catalog_label, 0) + \
                    scores_matrix[ind, k]

        return scores

    def predict_query(self, query, score_threshold=None):
        # Get img
        if type(query) in [str, np.string_]:
            query_img = utils.read_image(query, size=self.image_size)
        else:
            query_img = cv2.resize(query, (self.image_size, self.image_size))

        # Get keypoints
        query_keypoints = utils.get_keypoints(query_img,
                                              self.keypoint_stride,
                                              self.keypoint_sizes)

        # Get descriptors
        query_descriptors = utils.get_descriptors(query_img,
                                                  query_keypoints,
                                                  self.feature_extractor)

        # Get scores
        scores = self.get_query_scores(query_descriptors)

        return scores
