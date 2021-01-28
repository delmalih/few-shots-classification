# pylint: disable=too-many-arguments

##########################
# Imports
##########################


from typing import List

import numpy as np
from tensorflow import keras

from few_shots_clf import utils


##########################
# DataGenerator
##########################


class DataGenerator(keras.utils.Sequence):

    """[summary]
    """

    def __init__(
            self,
            img_paths: List,
            label_ids: List,
            image_size: int,
            batch_size: int,
            shuffle: bool = True):
        """ Initialisation """
        self.img_paths = img_paths
        self.label_ids = label_ids
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_paths_tmp: List, label_ids_tmp: List) -> (np.array, np.array):
        """ Generates data containing batch_size samples """
        # Initialization
        data_x = np.empty(
            (self.batch_size, self.image_size, self.image_size, 3))
        data_y = np.empty((self.batch_size))

        # Generate data
        for i, img_path in enumerate(img_paths_tmp):
            data_x[i] = utils.read_image(img_path, size=self.image_size) / 256.
            data_y[i] = label_ids_tmp[i]

        return data_x, data_y

    def __len__(self) -> int:
        """ Denotes the number of batches per epoch """
        return int(np.floor(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index: int) -> (np.array, np.array):
        """ Generate one batch of data """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        img_paths_tmp = [self.img_paths[k] for k in indexes]
        label_ids_tmp = [self.label_ids[k] for k in indexes]
        data_x, data_y = self.__data_generation(img_paths_tmp, label_ids_tmp)
        return data_x, data_y
