# pylint: disable=arguments-differ, too-many-ancestors

##########################
# Imports
##########################


import tensorflow as tf
from tensorflow import keras


##########################
# TripletModel
##########################


class TripletModel(keras.Model):

    """[summary]
    """

    def __init__(self, embedding_size: int):
        super().__init__()
        self.block1 = TripletBlock(32)
        self.block2 = TripletBlock(64)
        self.block3 = TripletBlock(128)
        self.flatten = keras.layers.Flatten()
        self.embedding_layer = keras.layers.Dense(embedding_size)

    def call(self, inputs: tf.Tensor):
        block_output = self.block1(inputs)
        block_output = self.block2(block_output)
        block_output = self.block3(block_output)
        flatten_output = self.flatten(block_output)
        embedding = self.embedding_layer(flatten_output)
        return embedding

    def get_config(self):
        return {
            "block1": self.block1,
            "block2": self.block2,
            "block3": self.block3,
            "flatten": self.flatten,
            "embedding_layer": self.embedding_layer,
        }


##########################
# TripletBlock
##########################


class TripletBlock(keras.layers.Layer):

    """[summary]
    """

    def __init__(self, n_filters: int, filters_size: int = 3):
        super().__init__()
        self.res_block1 = TripletResBlock(n_filters, filters_size)
        self.res_block2 = TripletResBlock(n_filters, filters_size)
        self.maxpool_layer = keras.layers.MaxPool2D()

    def call(self, inputs: tf.Tensor):
        res_block_output = self.res_block1(inputs)
        res_block_output = self.res_block2(res_block_output)
        output = self.maxpool_layer(res_block_output)
        return output


##########################
# TripletResBlock
##########################


class TripletResBlock(keras.layers.Layer):

    """[summary]
    """

    def __init__(self, n_filters: int, filters_size: int = 3):
        super().__init__()
        self.bottleneck_layer = TripletConvLayer(n_filters, filters_size=1)
        self.conv_bn_relu1 = TripletConvLayer(n_filters, filters_size)
        self.conv_bn_relu2 = TripletConvLayer(n_filters, filters_size)
        self.add_layer = keras.layers.Add()

    def call(self, inputs: tf.Tensor):
        bottleneck_output = self.bottleneck_layer(inputs)
        conv_bn_relu_output = self.conv_bn_relu1(bottleneck_output)
        conv_bn_relu_output = self.conv_bn_relu2(conv_bn_relu_output)
        output = self.add_layer([conv_bn_relu_output, bottleneck_output])
        return output


##########################
# TripletConvLayer
##########################


class TripletConvLayer(keras.layers.Layer):

    """[summary]
    """

    def __init__(self, n_filters: int, filters_size: int = 3):
        super().__init__()
        self.conv = keras.layers.Conv2D(n_filters,
                                        filters_size,
                                        padding="same")
        self.batch_norm = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()

    def call(self, inputs: tf.Tensor):
        conv_output = self.conv(inputs)
        batch_norm_output = self.batch_norm(conv_output)
        output = self.relu(batch_norm_output)
        return output
