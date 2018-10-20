"""
This module completes the neuro evolution

Property:
- index of generation
- number of individuals per generation (should be divided by 2)
- list of individuals per generation
- number of elites
- list of elites per generation
- max number of generations -> evolution termination
- Q value threshold -> evolution termination

Methods:
- initiate the evolution process
- sampling individuals for next generation
  - network mutation (same as later)
  - layer mutation   (same as later)
- evaluate individual fitness
- select top lambda elites
- neuro evolution and generate individuals for next generation
  - network mutation
  - layer mutation
  - network crossover
    - compute network similarity
    - pair networks for crossover
- evaluate individual fitness (repeat the loop)

"""
import os
import neuralnet as nn
from queue import Queue
from Family import family
from random import randint, sample, uniform, random


class NeuroEvolution(object):
    list_new_child = []
    list_bad_child = []
    list_family = []
    list_q = []
    family_num = 0
    now_family_num = 0
    now_rank = 0

    def __init__(self, population_size=10, max_gen=10):
        self.train_model_num = 0
        self.population_size = population_size
        self.max_gen = max_gen

    def create_base_family(self, father=None, mother=None):
        seed_father = self.create_seed_net(0, load_network=father)
        seed_mother = self.create_seed_net(1, load_network=mother)
        seed_family = family(self.family_num, seed_father, seed_mother, rank=1)
        seed_family.child_num = 1
        self.list_family.append(seed_family)

    def create_seed_net(self, num, net_name=None, load_network=None):
        seed_net = nn.NeuralNet(0, num, net_name=net_name, load_network=load_network, create_seed=True, is_seed=True)
        seed_net.kick_simulation()
        seed_net.output_deploy_network()
        return seed_net

    def out_list_family_info(self):
        print('the num of family: ' + str(len(self.list_family)))
        for now_family in self.list_family:
            print(now_family.out_info())

    def out_list_new_child_info(self):
        print('the num of good child list: ' + str(len(self.list_new_child)))
        for child in self.list_new_child:
            child.get_fnum_gen()

    def output(self, message):
        print(message)

    def output_family_info(self, family):
        print('--------------------Family' + str(family.family_num) + '--------------------')
        family_h = 'Family ' + str(family.family_num)
        family_f = family_h + ' father : net ' + family.father.get_fnum_gen()
        family_m = family_h + ' mother : net ' + family.mother.get_fnum_gen()
        family_q = family_h + ' Q : ' + str(family.Q)
        family_r = family_h + ' Rank : ' + str(family.rank)
        self.output(family_f)
        self.output(family_m)
        self.output(family_q)
        self.output(family_r)

    def run_family(self):
        # self.out_list_family_info()
        if len(self.list_family) == 0:
            print('have no family!!!!')
            return False
        now_family = self.list_family.pop()
        now_family.out_log()
        self.output_family_info(now_family)
        good_childs, bad_childs = now_family.create_new_childs()
        self.now_family_num = now_family.family_num
        self.now_rank = now_family.rank
        self.output('list of good childs:')
        for child in good_childs:
            child_gen_q = child.get_fnum_gen() + ':' + str(child.q_value)
            self.output(child_gen_q)
            self.list_new_child.append(child)
        if self.now_rank >= 1:
            self.output('list of bad childs:')
            for child in bad_childs:
                child_gen_q = child.get_fnum_gen() + ':' + str(child.q_value)
                self.output(child_gen_q)
                self.list_bad_child.append(child)
        return True

    def create_new_family(self):
        # self.out_list_new_child_info()
        self.out_list_family_info()
        print('----------Create new family----------')
        while (len(self.list_new_child) >= 2):
            father = self.list_new_child.pop()
            mother = self.list_new_child.pop()
            self.family_num += 1
            new_family = family(self.family_num, father, mother, rank=self.now_rank + 1)
            self.list_family.append(new_family)
            print(new_family.out_info())
        if len(self.list_family) == 0:
            while len(self.list_bad_child) >= 2:
                father = self.list_bad_child.pop()
                mother = self.list_bad_child.pop()
                self.family_num += 1
                new_family = family(self.family_num, father, mother, rank=self.now_rank - 1)
                self.list_family.append(new_family)
                print(new_family.out_info())
        print('-------------------------------------')

    def check_termination(self):
        if self.now_family_num < self.max_gen:
            return True
        else:
            return False
