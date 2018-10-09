import os
import numpy as np
import neuralnet as nn
import Layers
from random import randint, sample, uniform, random
from h5combine import combine

test = False


class family(object):
    child_num = 0
    list_child = []
    list_good_childs = []
    list_bad_childs = []

    def __init__(self, family_num_in, father, mother, rank=1, lifetime=20):
        self.father = father
        self.mother = mother
        self.rank = rank
        self.Q = max(father.q_value, mother.q_value)
        self.family_num = family_num_in
        self.lifetime = lifetime

    def out_info(self):
        print('Family' + str(self.family_num) + ':' + 'Rank ' + str(
            self.rank) + ' father ' + self.father.get_fnum_gen() + ',mother ' + self.mother.get_fnum_gen())

    def life_time(self, cn):
        return self.rank * 10 - cn
        # return self.lifetime - cn

    def check_family_lifetime(self):
        if self.life_time(self.child_num) >= 1:
            return True
        else:
            return False

    def add_new_child(self, new_child):
        self.list_child.append(new_child)
        new_child.output_deploy_network()

    def select_one_parent(self, layer, layer_a, layer_b):
        # check layer is closer to a or b
        # return True if a is selected, False if b
        diff_a = layer - layer_a
        diff_b = layer - layer_b
        if (sum(diff_a ** 2) <= sum(diff_b ** 2)):
            return True
        else:
            return False

    def select_layer_from_parent(self, net, warp_index):
        # pick the corresponding layer from the network
        # note that the layer index is the warped index, so maybe beyond the network depth
        # in this case, should use interpolation
        layer = net.select_layer_via_warp_index(warp_index)
        return layer

    def cross_over(self):
        depth_a = self.father.get_network_depth()
        depth_b = self.mother.get_network_depth()
        if (depth_a > depth_b):
            warp_length = depth_a
        else:
            warp_length = depth_b
        self.father.warp_layer_probablity(warp_length)
        self.mother.warp_layer_probablity(warp_length)
        layer_warp_a = self.father.get_warp_probability()
        layer_warp_b = self.mother.get_warp_probability()
        layer_output = (layer_warp_a + layer_warp_b) / 2
        new_list_layers = []
        for col_ind, layer_prob in enumerate(layer_output.T):
            layer_prob_a = layer_warp_a[:, col_ind]
            layer_prob_b = layer_warp_b[:, col_ind]
            select_a = self.select_one_parent(layer_prob, layer_prob_a, layer_prob_b)
            if select_a == True:
                layer = self.select_layer_from_parent(self.father, col_ind)
            else:
                layer = self.select_layer_from_parent(self.mother, col_ind)
            new_list_layers.append(layer)
        new_net = nn.NeuralNet(self.family_num, self.child_num)
        new_net.set_list_layers(new_list_layers, max(self.father.layer_num, self.mother.layer_num))
        if not test:
            c = combine(self.father, self.mother, new_net)
            c.combine()
        return new_net

    def cross_to_new_net_point(self):
        if not test:
            print('net ' + self.father.get_fnum_gen() + ' and net ' + self.mother.get_fnum_gen() + ' cross new net')
        cross_point = randint(1, self.father.get_network_depth() - 1)
        new_layer_list = self.father.list_layers[:cross_point] + self.mother.list_layers[cross_point:]
        new_net = nn.NeuralNet(self.family_num, self.child_num)
        new_net.set_list_layers(new_layer_list, max(self.father.layer_num, self.mother.layer_num))
        if not test:
            c = combine(self.father, self.mother, new_net)
            c.combine()
        return new_net

    def mutation_inter_layer(self, net):
        if not test:
            print('net ' + net.get_fnum_gen() + ' inter mutation')
        mutated_net = nn.NeuralNet(self.family_num, self.child_num)
        mutated_net.set_list_layers(net.get_list_layers(), net.layer_num)
        mutated_net.adjust_layer_list_random()
        return mutated_net

    def mutation_intra_layer(self, net):
        if not test:
            print('net ' + net.get_fnum_gen() + ' intra mutation')
        mutated_net = nn.NeuralNet(self.family_num, self.child_num)
        mutated_net.set_list_layers(net.get_list_layers(), net.layer_num)
        diff = randint(int(len(net.list_layers) / 5), int(len(net.list_layers) / 2))
        for _ in range(diff):
            mutated_net.adjust_layer_attr_random()
        return mutated_net

    def create_new_childs(self):
        self.list_good_childs.clear()
        self.list_bad_childs.clear()
        while (self.check_family_lifetime()):
            self.child_num += 1
            new_child = self.cross_over()
            if randint(0, 1) == 1:
                new_child = self.mutation_inter_layer(new_child)
            if randint(0, 1) == 1:
                new_child = self.mutation_intra_layer(new_child)
            self.list_child.append(new_child)
            new_child.father = new_child.ind_gen
            new_child.kick_simulation()
            if new_child.q_value > 0.1:
                new_child.output_deploy_network()
                if new_child.q_value > self.Q:
                    print('net ' + new_child.get_fnum_gen() + ' better than ' + str(self.Q))
                    self.list_good_childs.append(new_child)
                else:
                    if self.rank > 1:
                        print('net ' + new_child.get_fnum_gen() + ' worse than ' + str(self.Q))
                        if len(self.list_bad_childs) > 2:
                            if new_child.q_value > self.list_bad_childs[0].q_value:
                                self.list_bad_childs[0] = new_child
                            elif new_child.q_value > self.list_bad_childs[1].q_value:
                                self.list_bad_childs[1] = new_child
                        else:
                            self.list_bad_childs.append(new_child)
            else:
                self.child_num -= 1
        return self.list_good_childs, self.list_bad_childs
