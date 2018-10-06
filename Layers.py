"""
This module defines a set of layers to be integrated in the network.

                 hyperparameter
Convolution   |  number of output (or channels, or how many filters)
              |  kernel width
              |  stride (default: 1)
Inner Product |  number of output
Pooling       |  kernel width
              |  stride (default: 1)
-----------------------------------------------
ReLU          |  None  
Data (Test)   |  Input Dimension
Data (Train)  |  Input Dimension and Label Dimension
Loss          |  None (Euclidean Type)

Abstract Layer Method:
- output as string, to be called when network outputs

Layer specific methods and property please refer to each layer type
"""

# from caffe import layers as cl
import numpy as np
import keras
from random import randint


class Layer(object):
    # a vector of four dimension to represent four types of layers
    # Convolution:    [1.0, 0.0, 0.0, 0.0]
    # Inner Product:  [0.0, 1.0, 0.0, 0.0]
    # Pooling:        [0.0, 0.0, 1.0, 0.0]
    # ReLU:           [0.0, 0.0, 0.0, 1.0]
    # layer_probablity = np.array([0.0, 0.0, 0.0, 0.0])

    def __init__(self):
        self.layer_type = 'None'
        self.attr_dict = []

    def get_layer(self):
        pass

    def set_name(self, net_index, layer_index, name=None):
        if name is not None:
            self.name = name
        else:
            self.name = net_index + '_' + self.layer_type + '_' + str(layer_index)

    def get_name(self):
        return self.name

    def get_template_file_name(self):
        template_file_name = 'templates/layers/' + self.layer_type + '.txt'
        return template_file_name

    def construct_dict(self):
        raise NotImplementedError("Must override construct_dict in child class")

    def adjust_attr_random(self, intra=False, attr_list=None):
        if attr_list == None:
            attr_keys = self.attr_dict.keys()
        for key in attr_keys:
            num_options = len(self.attr_options_dict[key])
            value_option = randint(0, num_options - 1)
            value = self.attr_options_dict[key][value_option]
            if intra == True:
                print(key + ':' + str(self.attr_dict[key]) + '->' + str(value))
            self.attr_dict[key] = value

    def output(self, bottom_layer=None):
        template_file_name = self.get_template_file_name()
        with open(template_file_name, 'r') as template_file:
            template_str = template_file.read()
        info = self.construct_dict(bottom_layer)
        out_str = template_str.format(**info)
        # out_str = out_str.replace('[', '{')
        # out_str = out_str.replace(']', '}')
        return out_str

    def print_attr(self):
        print(self.name)
        for key in self.attr_dict.keys():
            print(key + ':' + str(self.attr_dict[key]))


class ConvLayer(Layer):
    """Convolution Layer
    
    Attributes:
    - number of output
    - kernel width
    - stride (default value: 1)
    - max number of output
    
    """
    layer_type = 'convolution'
    layer_probablity = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    attr_options_dict = {"num_output": [16, 32, 64, 128, 256, 512],
                         "kernel_width": [4, 6, 8, 10, 12, 14, 16],
                         "stride": [1, 2, 4]}

    def __init__(self, num_output_in=None, kernel_width_in=None, stride_in=1, padding_in='valid', use_bias_in=True,
                 name_in=None, concatenate=None, ):
        if num_output_in == None and kernel_width_in == None:
            self.attr_dict = {"num_output": None,
                              "kernel_width": None,
                              "stride": None,
                              }
            self.adjust_attr_random()
        else:
            self.attr_dict = {"num_output": int(num_output_in),
                              "kernel_width": int(kernel_width_in),
                              "stride": int(stride_in),
                              }
        self.padding = padding_in
        self.use_bias = use_bias_in
        self.concatenate = concatenate
        if name_in != None:
            self.name = name_in

    def construct_dict(self, bottom_layer):
        info = {'name': self.name,
                'bottom': bottom_layer,
                'top': self.name,
                'num_output': self.attr_dict["num_output"],
                'kernel_w': self.attr_dict["kernel_width"],
                'stride': self.attr_dict["stride"],
                'padding': self.padding,
                'use_bias': self.use_bias,
                'concatenate': self.concatenate
                }
        return info

    def get_layer(self):
        return keras.layers.Conv2D(filters=self.attr_dict["num_output"],
                                   kernel_size=self.attr_dict["kernel_width"],
                                   strides=self.attr_dict["stride"],
                                   padding=self.padding,
                                   use_bias=self.use_bias,
                                   name=self.name)


class InnerLayer(Layer):
    """Inner Product Layer
    
    Attributes:
    - number of output
    - max number of output
    """
    layer_type = 'innerproduct'
    layer_probablity = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    attr_options_dict = {"num_output": [16, 32, 64, 128, 256, 512]}

    def __init__(self, num_output_in=None, name_in=None, concatenate=None):
        if num_output_in == None:
            self.attr_dict = {"num_output": None}
            self.adjust_attr_random()
        else:
            self.attr_dict = {"num_output": int(num_output_in)}
        self.concatenate = concatenate
        if name_in != None:
            self.name = name_in

    def construct_dict(self, bottom_layer):
        info = {'name': self.name,
                'bottom': bottom_layer,
                'top': self.name,
                'num_output': self.attr_dict["num_output"],
                'concatenate': self.concatenate,
                }
        return info

    def get_layer(self):
        return keras.layers.Dense(self.attr_dict["num_output"],
                                  name=self.name)


class PoolLayer(Layer):
    """Pooling layer
    
    Attributes:
    - kernel width
    - stride (default value: 1)
    """
    layer_type = 'pool'
    layer_probablity = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    attr_options_dict = {"kernel_size": [4, 6, 8, 10, 12, 14, 16],
                         "strides": [1, 2],
                         "pool_type": ['max', 'average', 'globalaverage']}

    def __init__(self, kernel_size_in=None, strides_in=None, pool_type_in=None, name_in=None, concatenate=None, ):
        if kernel_size_in == None and strides_in == None and pool_type_in == None:
            self.attr_dict = {"kernel_size": None,
                              "strides": None,
                              "pool_type": None}
            self.adjust_attr_random()
        else:
            self.attr_dict = {"kernel_size": int(kernel_size_in),
                              "strides": int(strides_in),
                              "pool_type": pool_type_in}
        self.concatenate = concatenate
        if name_in != None:
            self.name = name_in

    def construct_dict(self, bottom_layer):
        info = {'name': self.name,
                'bottom': bottom_layer,
                'top': self.name,
                'kernel_size': self.attr_dict["kernel_size"],
                'strides': self.attr_dict["strides"],
                'pool_type': self.attr_dict["pool_type"],
                'concatenate': self.concatenate,
                }
        return info

    def get_layer(self):
        if self.attr_dict["pool_type"] == 'max':
            return keras.layers.MaxPooling2D(pool_size=self.attr_dict["kernel_size"],
                                             strides=(self.attr_dict["strides"], self.attr_dict["strides"]),
                                             name=self.name)
        elif self.attr_dict["pool_type"] == 'average':
            return keras.layers.AveragePooling2D(pool_size=self.attr_dict["kernel_size"],
                                                 strides=(self.attr_dict["strides"], self.attr_dict["strides"]),
                                                 name=self.name)
        elif self.attr_dict["pool_type"] == 'globalaverage':
            return keras.layers.GlobalAveragePooling2D()


class ActivationLayer(Layer):
    layer_type = 'activation'
    layer_probablity = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    attr_options_dict = {"active_type": ['relu', 'sigmoid', 'softmax']}

    def __init__(self, active_type_in='relu', name_in=None, concatenate=None):
        self.attr_dict = {'active_type': active_type_in}
        if name_in != None:
            self.name = name_in
        self.concatenate = concatenate

    def construct_dict(self, bottom_layer):
        info = {'name': self.name,
                'bottom': bottom_layer,
                'top': self.name,
                'active_type': self.attr_dict["active_type"],
                'concatenate': self.concatenate,
                }
        return info

    def get_layer(self):
        return keras.layers.Activation(self.attr_dict["active_type"], name=self.name)


class BatchNormalizationLayer(Layer):
    layer_type = 'batchnormalization'
    layer_probablity = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def __init__(self, name_in=None, concatenate=None):
        self.attr_dict = {}
        if name_in != None:
            self.name = name_in
        self.concatenate = concatenate

    def construct_dict(self, bottom_layer):
        info = {'name': self.name,
                'bottom': bottom_layer,
                'top': self.name,
                'concatenate': self.concatenate,
                }
        return info

    def get_layer(self):
        return keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=self.name)


class ConcatenateLayer(Layer):
    layer_type = 'concatenate'
    layer_probablity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def __init__(self, name_in=None, concatenate=None):
        self.attr_dict = {}
        if name_in != None:
            self.name = name_in
        self.concatenate = concatenate

    def construct_dict(self, bottom_layer):
        info = {'name': self,
                'bottom': bottom_layer,
                'top': self.name,
                'concatenate': self.concatenate,
                }
        return info

    def get_layer(self):
        return keras.layers.Concatenate(axis=3, name=self.name)


class ZeroPaddingLayer(Layer):
    layer_type = 'zeropadding'
    layer_probablity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def __init__(self, padding_in=1, name_in=None, concatenate=None):
        self.attr_dict = {}
        if name_in != None:
            self.name = name_in
        self.padding = padding_in
        self.concatenate = concatenate

    def construct_dict(self, bottom_layer):
        info = {'name': self.name,
                'bottom': bottom_layer,
                'top': self.name,
                'padding': self.padding,
                'concatenate': self.concatenate,
                }
        return info

    def get_layer(self):
        return keras.layers.ZeroPadding2D(padding=((self.padding, self.padding), (self.padding, self.padding)),
                                          name=self.name)
