"""
This module defines the class of neural net.

Property:
- list of layers
- list of layer probablity
- max number of layer

Methods:
- construct a network from a proto file
- output the network to a proto file
- add/delete a layer from a net
- adjust the layer hyperparameter by calling the member function of a layer
- warping a layer to a specified length
- run the simulation, and collect the fitness value
"""

import numpy as np
from os.path import join
from Fcifar100 import Fitness
from random import randint
from Layers import *

test = False


class NeuralNet(object):
    dynamic_layers = [ConvLayer, InnerLayer, PoolLayer, ActivationLayer, BatchNormalizationLayer, ZeroPaddingLayer]

    def __init__(self, family_num_in, ind_gen_in, is_seed=False, net_name=None, load_network=None,
                 create_seed=False):
        import os

        self.list_layers = []
        self.layer_probablity_warp = []
        self.father = None
        self.q_value = None
        self.layer_num = 0

        self.family_num = family_num_in
        self.ind_gen = ind_gen_in
        if load_network != None:
            self.load_network(load_network, create_seed=create_seed)
        elif ind_gen_in == 0 or is_seed:
            self.create_seed_net(net_name)
            self.father = 'base'
        else:
            self.create_empty_net()
        self.folder_name = self.get_folder_name()
        try:
            os.mkdir(self.folder_name)
        except:
            pass

    def rank(self):
        return int(self.q_value*10)+1

    def set_father(self, father):
        self.father = father

    def get_network_depth(self):
        return len(self.list_layers)

    def get_warp_probability(self):
        if len(self.layer_probablity_warp) == 0:
            raise "Layer probablity matrix after warping is not initialized"
        return self.layer_probablity_warp

    def get_fnum_gen(self):
        return str(self.family_num) + '_' + str(self.ind_gen)

    def get_folder_name(self):
        return "population/" + self.get_fnum_gen() + '/'

    def get_network_name(self, is_deploy):
        if is_deploy:
            return "actor_deploy_" + str(self.ind_gen)
        else:
            return "actor_solver_" + str(self.ind_gen)

    def get_list_layers(self):
        # make a copy, so this internal doesn't get affected
        import copy as cp
        return cp.deepcopy(self.list_layers)

    def set_list_layers(self, list_layers_in, layer_num_in):
        self.layer_num = layer_num_in
        self.list_layers = list_layers_in

    def create_seed_net(self, net_name=None):
        if net_name == None:
            self.list_layers = []
            self.append_layer(ConvLayer(32, 4, 1))
            self.append_layer(ActivationLayer('relu'))
            self.append_layer(ConvLayer(32, 4, 2))
            self.append_layer(ActivationLayer('relu'))
            self.append_layer(ConvLayer(32, 4, 2))
            self.append_layer(ActivationLayer('relu'))
            self.append_layer(InnerLayer(64))
            self.append_layer(ActivationLayer('relu'))
        elif net_name == 'DN169':
            self.list_layers = []
            self.append_layer(ZeroPaddingLayer(3))
            self.append_layer(ConvLayer(64, 7, 2, use_bias_in=False))
            self.append_layer(BatchNormalizationLayer())
            self.append_layer(ActivationLayer('relu'))
            self.append_layer(ZeroPaddingLayer(1))
            self.append_layer(PoolLayer(3, 2, type_in='max', concatenate=True))
            self.dense_block(6)
            self.transition_block(128)
            self.dense_block(12)
            self.transition_block(256)
            self.dense_block(32)
            self.transition_block(640)
            self.dense_block(32)
            self.append_layer(BatchNormalizationLayer())
            self.append_layer(ActivationLayer('softmax'))
        else:
            raise "No this net seed"

    def append_layer(self, layer_in):
        self.layer_num = self.layer_num + 1
        layer_in.set_name(self.get_fnum_gen(), self.layer_num)
        self.list_layers.append(layer_in)

    def dense_block(self, blocks):
        for i in range(blocks):
            self.append_layer(BatchNormalizationLayer())
            self.append_layer(ActivationLayer('relu'))
            self.append_layer(ConvLayer(4 * 32, 1, 1, use_bias_in=False))
            self.append_layer(BatchNormalizationLayer())
            self.append_layer(ActivationLayer('relu'))
            self.append_layer(ConvLayer(32, 3, 1, padding_in='same', use_bias_in=False))
            self.append_layer(ConcatenateLayer(concatenate=True))

    def transition_block(self, conv_in):
        self.append_layer(BatchNormalizationLayer())
        self.append_layer(ActivationLayer('relu'))
        self.append_layer(ConvLayer(conv_in, 1, 1, use_bias_in=False))
        self.append_layer(PoolLayer(2, 2, pool_type_in='average', concatenate=True))

    def create_empty_net(self):
        pass

    def select_layer_via_warp_index(self, warp_index):
        if len(self.layer_probablity_warp) == 0:
            raise "Layer probablity matrix after warping is not initialized"
        rel_pose = warp_index / self.layer_probablity_warp.shape[1] * len(self.list_layers)
        pose = int(round(rel_pose))
        if pose > (len(self.list_layers) - 1):
            pose = len(self.list_layers) - 1
        return self.list_layers[pose]

    def insert_layer(self, layer, layer_index):
        if (layer_index >= len(self.list_layers)):
            raise "the inserted position is beyond the current depth of the network"
        self.list_layers.insert(layer_index, layer)

    def delete_layer(self, layer_index):
        if not test:
            print('delete layer:')
            self.list_layers[layer_index].print_attr()
        del self.list_layers[layer_index]

    def adjust_layer_list_random(self):
        diff_num_layers = 0
        while diff_num_layers == 0:
            diff_num_layers = randint(-int(len(self.list_layers) / 5), int(len(self.list_layers)))
        if (diff_num_layers > 0):
            while (diff_num_layers != 0):
                self.insert_layer_random_position()
                diff_num_layers = diff_num_layers - 1
        else:
            while (diff_num_layers != 0):
                self.delete_layer_random_position()
                diff_num_layers = diff_num_layers + 1

    def insert_layer_random_position(self):
        layer = self.generate_random_layer()
        layer_index = randint(0, len(self.list_layers) - 1)
        self.layer_num = self.layer_num + 1
        layer.set_name(self.get_fnum_gen(), self.layer_num)
        self.insert_layer(layer, layer_index)
        if not test:
            print('insert layer')
            layer.print_attr()

    def generate_random_layer(self):
        layer_type = randint(0, len(self.dynamic_layers) - 1)
        layer = self.dynamic_layers[layer_type]()
        return layer

    def delete_layer_random_position(self):
        layer_index = randint(0, len(self.list_layers) - 1)
        self.delete_layer(layer_index)

    def adjust_layer_attr_random(self, layer_index=None):
        # layer = self.list_layers[layer_index]
        while True:
            layer_index = randint(0, self.get_network_depth() - 1)
            layer = self.list_layers[layer_index]
            if type(layer) in self.dynamic_layers:
                break
        layer.adjust_attr_random(intra=not test)
        self.layer_num = self.layer_num + 1
        layer.set_name(self.get_fnum_gen(), self.layer_num)
        if not test:
            layer.print_attr()

    def kick_simulation(self):
        import gc
        from datetime import datetime
        trained = False
        if self.q_value == None:
            print("net " + self.get_fnum_gen() + " train")
            if test:
                self.q_value = float(randint(0, 10000)) / 10000
            else:
                f = Fitness(self.list_layers, self.father, self.folder_name)
                self.q_value = f.train()
                del f
                gc.collect()
            print("net " + self.get_fnum_gen() + " : " + str(self.q_value))
            if self.q_value != 0:
                f = open('log.txt', 'a')
                f.write(datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z') + '   ')
                f.write(self.get_fnum_gen() + '   ')
                f.write(str(self.q_value) + '\n')
                f.close()
                trained = True
        return self.q_value, trained

    def warp_layer_probablity(self, warp_length):
        x = np.linspace(0, len(self.list_layers) - 1, warp_length)
        xp = np.linspace(0, len(self.list_layers) - 1, len(self.list_layers))
        yp_mat = self.construct_layer_probability_matrix()
        yp_mat_interp = self.interpolate_layer_probability_matrix(x, xp, yp_mat)
        self.layer_probablity_warp = yp_mat_interp
        return yp_mat_interp

    def construct_layer_probability_matrix(self):
        prob_list = []
        for layer in self.list_layers:
            prob_list.append(layer.layer_probablity)
        prob_mat = np.array(prob_list)
        self.layer_probablity = np.transpose(prob_mat)
        return self.layer_probablity

    def interpolate_layer_probability_matrix(self, x, xp, yp):
        from scipy.interpolate import interp1d
        f = interp1d(xp, yp)
        interp_mat = f(x)
        return interp_mat

    def output_deploy_network(self):
        import os
        output_file_name = self.folder_name + "/actor.txt"
        output_str = ""
        for layer_index, layer in enumerate(self.list_layers):
            if layer_index == 0:
                bottom_layer = "data"
            else:
                bottom_layer = self.list_layers[layer_index - 1].get_name()
            output_str = output_str + layer.output(bottom_layer)
        with open(output_file_name, "w") as out_file:
            out_file.write(output_str)
        score_file_name = os.path.join(self.folder_name, 'score.txt')
        score_str = str(self.q_value)
        with open(score_file_name, "w") as out_score_file:
            out_score_file.write(score_str)
        layer_num_file = os.path.join(self.folder_name, 'layer_num.txt')
        layer_num_str = str(self.layer_num)
        with open(layer_num_file, "w") as out_layer_num_file:
            out_layer_num_file.write(layer_num_str)

    def load_network(self, father, create_seed=False):
        if create_seed:
            file_path = join('seed', str(father))
        else:
            file_path = join('population', str(father))
            score = join(file_path, 'score.txt')
            with open(score, 'r') as read_score:
                score_str = read_score.read()
                self.q_value = max(float(score_str), 0)
            layer_num_path = join(file_path, 'layer_num.txt')
            with open(layer_num_path, 'r') as read_layer_num:
                layer_num_str = read_layer_num.read()
                self.layer_num = int(layer_num_str)
        actor = join(file_path, 'actor.txt')
        self.father = father
        new_layer = False
        layer_dict = {}
        with open(actor, 'r') as read_file:
            for line in read_file.readlines():
                if 'layer {' in line:
                    new_layer = True
                    layer_dict.clear()
                    continue
                if '}' in line:
                    new_layer = False
                    self.layer_num += 1
                    if layer_dict['type'] == 'Convolution':
                        self.list_layers.append(ConvLayer(num_output_in=layer_dict['num_output'],
                                                          kernel_width_in=layer_dict['kernel_w'],
                                                          stride_in=layer_dict['stride'],
                                                          padding_in=layer_dict['padding'],
                                                          use_bias_in=layer_dict['use_bias'],
                                                          name_in=layer_dict['name'],
                                                          concatenate=layer_dict['concatenate'], ))
                    elif layer_dict['type'] == 'Activation':
                        self.list_layers.append(ActivationLayer(active_type_in=layer_dict['active_type'],
                                                                name_in=layer_dict['name'],
                                                                concatenate=layer_dict['concatenate']))
                    elif layer_dict['type'] == 'InnerProduct':
                        self.list_layers.append(InnerLayer(num_output_in=layer_dict['num_output'],
                                                           name_in=layer_dict['name'],
                                                           concatenate=layer_dict['concatenate']))
                    elif layer_dict['type'] == 'Pooling':
                        self.list_layers.append(PoolLayer(kernel_size_in=layer_dict['kernel_size'],
                                                          strides_in=layer_dict['stride'],
                                                          pool_type_in=layer_dict['pool_type'],
                                                          name_in=layer_dict['name'],
                                                          concatenate=layer_dict['concatenate']))
                    elif layer_dict['type'] == 'ZeroPadding':
                        self.list_layers.append(ZeroPaddingLayer(padding_in=layer_dict['padding'],
                                                                 name_in=layer_dict['name'],
                                                                 concatenate=layer_dict['concatenate']))
                    elif layer_dict['type'] == 'BatchNormalization':
                        self.list_layers.append(BatchNormalizationLayer(name_in=layer_dict['name'],
                                                                        concatenate=layer_dict['concatenate']))
                    elif layer_dict['type'] == 'Concatenate':
                        self.list_layers.append(ConcatenateLayer(name_in=layer_dict['name'],
                                                                 concatenate=layer_dict['concatenate']))
                    # print(layer_dict)
                if new_layer:
                    key = line.strip().split(':')[0]
                    value = line.strip().split(':')[1]
                    layer_dict[key] = value

                # print(line.strip())


if __name__ == '__main__':
    nn = NeuralNet(0, 0, load_network='base')
