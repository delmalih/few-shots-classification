# pylint: disable=pointless-string-statement

from .fm_classifier import FMClassifier
from .bow_classifier import BOWClassifier

""" Top-level package for Few Shots Classification Library. """

# Read version
with open("./__version__.txt", "r") as version_file:
    version = version_file.read()

__author__ = "David El Malih"
__author_email__ = "da.elmalih@gmail.com"
__version__ = version
__description__ = "Few Shots Classification Library. " + \
                  "Classification task from a small amount of data."
