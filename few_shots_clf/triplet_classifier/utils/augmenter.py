##########################
# Imports
##########################


import numpy as np
from imgaug import augmenters as iaa


##########################
# Augmenter
##########################


class Augmenter:

    """[summary]
    """

    def __init__(self, augment_factor: int):
        self.augment_factor = augment_factor
        self.probability = 0.3
        self.seq = iaa.Sequential(
            [
                # horizontal flip
                iaa.Sometimes(self.probability, iaa.Fliplr(0.5)),
                # vertical flip
                iaa.Sometimes(self.probability, iaa.Flipud(0.5)),
                # scale
                iaa.Sometimes(self.probability, iaa.Affine(scale={"x": (0.8, 1.2),
                                                                  "y": (0.8, 1.2)})),
                # translate
                iaa.Sometimes(self.probability, iaa.Affine(translate_percent={"x": (-0.1, 0.1),
                                                                              "y": (-0.1, 0.1)})),
                # shear
                iaa.Sometimes(self.probability, iaa.Affine(shear=(-10, 10))),
                # blur
                iaa.Sometimes(self.probability, iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(3, 7)),
                    iaa.MedianBlur(k=(3, 7)),
                ])),
                # gaussian noise
                iaa.Sometimes(self.probability,
                              iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255))),
                # brightness
                iaa.Sometimes(self.probability,
                              iaa.Multiply((0.5, 1.5), per_channel=self.probability))
            ],
            random_order=True
        )

    def augment(self, images: np.array, labels: np.array) -> (np.array, np.array):
        """Perform data augmentation on given dataset

        Args:
            images (np.array): Original images
            labels (np.array): Original labels

        Returns:
            np.array: Augmented images
            np.array: Augmented labels
        """
        aug_images = aug_labels = None
        for _ in range(self.augment_factor):
            if aug_images is None:
                aug_images = self.seq(images=images)
                aug_labels = labels
            else:
                aug_images = np.concatenate((aug_images,
                                             self.seq(images=images)))
                aug_labels = np.concatenate((aug_labels,
                                             labels))
        return aug_images, aug_labels
