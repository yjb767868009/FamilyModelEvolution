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

    def __init__(self, family_num_in, father, mother, R=1):
        self.father = father
        self.mother = mother
        self.R = R
        self.Q = max(father.q_value, mother.q_value)
        self.family_num = family_num_in

    def life_time(self, cn):
        return 20 - cn

    def check_family_lifetime(self):
        if self.life_time(self.child_num) >= 1:
            return True
        else:
            return False

    def add_new_child(self, new_child):
        self.list_child.append(new_child)
        new_child.output_deploy_network()

    def cross_to_new_net_point(self):
        print('net ' + self.father.get_fnum_gen() + ' and net ' + self.mother.get_fnum_gen() + ' cross new net')
        cross_point = randint(1, self.father.get_network_depth() - 1)
        new_layer_list = self.father.list_layers[:cross_point] + self.mother.list_layers[cross_point:]
        new_net = nn.NeuralNet(self.family_num, self.child_num)
        new_net.set_list_layers(new_layer_list, max(self.father.layer_num, self.mother.layer_num))
        if not test:
            c = combine(self.father, self.mother, new_net, cross_point)
            c.combine()
        return new_net

    def mutation_inter_layer(self, net):
        print('net ' + net.get_fnum_gen() + ' inter mutation')
        mutated_net = nn.NeuralNet(self.family_num, self.child_num)
        mutated_net.set_list_layers(net.get_list_layers(), net.layer_num)
        mutated_net.adjust_layer_list_random()
        return mutated_net

    def mutation_intra_layer(self, net):
        print('net ' + net.get_fnum_gen() + ' intra mutation')
        mutated_net = nn.NeuralNet(self.family_num, self.child_num)
        mutated_net.set_list_layers(net.get_list_layers(), net.layer_num)
        diff = randint(int(len(net.list_layers) / 5), int(len(net.list_layers) / 2))
        for _ in range(diff):
            mutated_net.adjust_layer_attr_random()
        return mutated_net

    def create_new_childs(self):
        self.list_good_childs.clear()
        while (self.check_family_lifetime()):
            self.child_num += 1
            new_child = self.cross_to_new_net_point()
            if randint(0, 1) == 1:
                new_child = self.mutation_inter_layer(new_child)
            if randint(0, 1) == 1:
                new_child = self.mutation_intra_layer(new_child)
            self.list_child.append(new_child)
            new_child.kick_simulation()
            if new_child.q_value > 0.1:
                new_child.output_deploy_network()
                if new_child.q_value > self.Q:
                    print('net ' + new_child.get_fnum_gen() + ' better than ' + str(self.Q))
                    self.list_good_childs.append(new_child)
            else:
                self.child_num -= 1
        return self.list_good_childs
