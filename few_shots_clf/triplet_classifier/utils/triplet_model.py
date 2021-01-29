# pylint: disable=arguments-differ, too-many-ancestors

##########################
# Imports
##########################


import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


##########################
# TripletModel
##########################


class TripletModel(keras.Model):

    """[summary]
    """

    def __init__(self, embedding_size: int, model_backbone: str):
        super().__init__()
        self.backbone = self._get_backbone(model_backbone)
        self.flatten = keras.layers.Flatten(name="flatten_layer")
        self.embedding_layer = keras.layers.Dense(embedding_size,
                                                  name="embedding_layer")
        self.l2_normalization_layer = keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=1),
                                                          name="l2_normalization_layer")

    @staticmethod
    def _get_backbone(model_backbone: str) -> keras.Model:
        """[description]

        Args:
            model_backbone (str): [description]

        Returns:
            keras.Model: [description]
        """
        if model_backbone == "vgg16":
            return keras.applications.VGG16(include_top=False)
        if model_backbone == "vgg19":
            return keras.applications.VGG19(include_top=False)
        if model_backbone == "resnet50":
            return keras.applications.ResNet50(include_top=False)
        if model_backbone == "resnet101":
            return keras.applications.ResNet101(include_top=False)
        if model_backbone == "resnet152":
            return keras.applications.ResNet152(include_top=False)
        return keras.applications.ResNet101(include_top=False)

    def call(self, inputs: tf.Tensor):
        backbone_output = self.backbone(inputs)
        flatten_output = self.flatten(backbone_output)
        embedding = self.embedding_layer(flatten_output)
        embedding = self.l2_normalization_layer(embedding)
        return embedding

    def get_config(self):
        return {
            "backbone": self.backbone,
            "flatten": self.flatten,
            "embedding_layer": self.embedding_layer,
            "l2_normalization_layer": self.l2_normalization_layer,
        }
