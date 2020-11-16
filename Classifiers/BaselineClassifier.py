##########################
# Imports
##########################


# Global imports
import json
import numpy as np
from tqdm import tqdm

# Local imports
import utils
from utils import constants


##########################
# Classifier
##########################


class BaselineClassifier(object):
    def __init__(self, catalog_folder, params={}):
        self.catalog_folder = catalog_folder
        self.get_params(params)

    def get_params(self, params):
        # General
        self.verbose = params.get(
            "verbose",
            constants.VERBOSE)
        self.feature_extractor = params.get(
            "feature_extractor",
            constants.DEFAULT_FEATURE_EXTRACTOR)
        self.matcher_index_params = params.get(
            "matcher_index_params",
            constants.DEFAULT_MATCHER_INDEX_PARAMS)
        self.matcher_query_params = params.get(
            "matcher_query_params",
            constants.DEFAULT_MATCHER_QUERY_PARAMS)
        self.background_label = params.get(
            "background_label",
            constants.BACKGROUND_LABEL)
        self.image_size = params.get(
            "image_size",
            constants.DEFAULT_CLASSIFIER_IMAGE_SIZE)
        self.keypoint_stride = params.get(
            "keypoint_stride",
            constants.DEFAULT_CLASSIFIER_KEYPOINT_STRIDE)
        self.keypoint_sizes = params.get(
            "keypoint_sizes",
            constants.DEFAULT_CLASSIFIER_KEYPOINT_SIZES)

        # Classifier Custom
        self.matcher_path_custom = params.get(
            "matcher_path",
            constants.DEFAULT_CLASSIFIER_CUSTOM_MATCHER_PATH)
        self.force_matcher_compute = params.get(
            "force_matcher_compute",
            constants.DEFAULT_CLASSIFIER_CUSTOM_FORCE_MATCHER_COMPUTE)
        self.k_nn_custom = params.get(
            "k_nn",
            constants.DEFAULT_CLASSIFIER_CUSTOM_K_NN)
        self.score_sigma = params.get(
            "sigma",
            constants.DEFAULT_CLASSIFIER_CUSTOM_SCORE_SIGMA)

        # Classifier BoW
        self.vocab_size = params.get(
            "vocab_size",
            constants.DEFAULT_CLASSIFIER_BOW_VOCAB_SIZE)
        self.vocab_path = params.get(
            "vocab_path",
            constants.DEFAULT_CLASSIFIER_BOW_VOCAB_PATH)
        self.catalog_features_path = params.get(
            "catalog_features_path",
            constants.DEFAULT_CLASSIFIER_BOW_CATALOG_FEATURES_PATH)
        self.force_vocab_compute = params.get(
            "force_vocab_compute",
            constants.DEFAULT_CLASSIFIER_BOW_FORCE_VOCAB_COMPUTE)
        self.force_catalog_features_compute = params.get(
            "force_catalog_features_compute",
            constants.DEFAULT_CLASSIFIER_BOW_FORCE_CATALOG_FEATURES_COMPUTE)

    def predict_query(self, query):
        return {}

    def predict_query_batch(self, query_paths):
        # Init. results
        results = {}

        # Init. iterator
        iterator = utils.get_iterator(
            query_paths,
            self.verbose,
            desc="Query prediction")

        # Compute results
        for query_path in iterator:
            query_id = "/".join(query_path.split("/")[-2:])
            results[query_id] = self.predict_query(query_path)

        return results

    def compute_top_k_accuracy(self, predictions, k):
        # Init. counters
        correct_counter = {}
        total_counter = {}
        accuracy = {}

        # Loop
        for img_id in predictions.keys():
            # Get ground truth label
            gt_label = img_id.split("/")[0]

            # Get predicted labels
            pred_labels = sorted(predictions[img_id].keys(),
                                 key=lambda x: predictions[img_id][x],
                                 reverse=True)[:k]

            # Update counters
            if gt_label in pred_labels:
                correct_counter[gt_label] = correct_counter.get(
                    gt_label, 0) + 1
            total_counter[gt_label] = total_counter.get(gt_label, 0) + 1

        # Compute accuracy
        for label in total_counter:
            accuracy[label] = correct_counter[label] / total_counter[label]

        return accuracy

    def compute_metrics(self, query_path):
        # List query images
        query_paths = utils.get_all_files(query_path)

        # Compute predictions
        predictions = self.predict_query_batch(query_paths)

        # Compute metrics
        top1 = self.compute_top_k_accuracy(predictions, k=1)
        top3 = self.compute_top_k_accuracy(predictions, k=3)
        top5 = self.compute_top_k_accuracy(predictions, k=5)

        return top1, top3, top5
