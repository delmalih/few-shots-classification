##########################
# Imports
##########################


# Global
from setuptools import setup

# Local
import few_shots_clf


##########################
# Setup
##########################


requirements = [
    "tqdm",
    "autopep8",
    "pylint",
    "pytest",
    "nmslib",
    "imutils",
    "easydict",
    "numpy >= 1.15.4",
    "opencv-python == 3.4.2.16",
    "opencv-contrib-python == 3.4.2.16",
]


setup(
    name="few_shots_clf",
    version=few_shots_clf.__version__,
    author=few_shots_clf.__author__,
    author_email=few_shots_clf.__author_email__,
    description=few_shots_clf.__description__,
    packages=["few_shots_clf", "few_shots_clf.test"],
    url="http://pypi.python.org/pypi/few_shots_clf/",
    license="LICENSE.txt",
    long_description=open("README.md").read(),
    install_requires=requirements,
)
