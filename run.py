##########################
# Imports
##########################


# Global imports
import os
import argparse
from glob import glob

# Local imports
import utils

# Classifiers
from Classifiers.BaselineClassifier import BaselineClassifier
from Classifiers.CustomClassifier import CustomClassifier
# from Classifiers.BOWClassifier import BOWClassifier


##########################
# Functions
##########################


def parse_args():
    # Prepare parser
    parser = argparse.ArgumentParser(
        description="Arguments for running classifier")
    parser.add_argument(
        "-cf",
        "--catalog_folder",
        dest="catalog_folder",
        help="Path to catalog images folder",
        required=True)
    parser.add_argument(
        "-qf",
        "--query_folder",
        dest="query_folder",
        help="Path to query images folder")
    parser.add_argument(
        "-qi",
        "--query_image",
        dest="query_image",
        help="Path to query image")
    parser.add_argument(
        "-clf",
        "--classifier",
        dest="classifier",
        help="Classifier : Baseline, Custom (default), BOW",
        default="Custom")

    # Init parser
    args = parser.parse_args()

    return args


def get_classifier(args):
    if args.classifier == "Baseline":
        return BaselineClassifier(args.catalog_folder)
    elif args.classifier == "Custom":
        return CustomClassifier(args.catalog_folder)
    # elif args.classifier == "BOW":
    #     return BOWClassifier(args.catalog_folder)
    else:
        return CustomClassifier(args.catalog_folder)


##########################
# Main
##########################


if __name__ == "__main__":
    # Init. "files" folder
    if not os.path.exists("./files"):
        os.makedirs("./files")

    # Get args
    args = parse_args()

    # Init. classifier
    classifier = get_classifier(args)

    # Case single image
    if args.query_image:
        # Compute scores
        scores = classifier.predict_query(args.query_image)

        # Compute TOP 5 labels
        top5labels = sorted(scores.keys(),
                            key=lambda x: scores[x],
                            reverse=True)[:5]

        # Print
        print("Top 5 labels :")
        for k, label in enumerate(top5labels):
            print(f"{k + 1}. \
                  Label = {label}, \
                  Score = {scores[label]}")

    # Case multiple images
    if args.query_folder:
        # Compute metrics
        top1, top3, top5 = classifier.compute_metrics(args.query_folder)

        # Print
        print(f"Metrics: \
                Top1 = {top1 * 100:.3f}% | \
                Top3 = {top3 * 100:.3f}% | \
                Top5 = {top5 * 100:.3f}%")
