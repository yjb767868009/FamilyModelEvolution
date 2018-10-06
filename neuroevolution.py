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
    queue_new_child = Queue()
    queue_family = Queue()
    list_q = []
    family_num = 0

    def __init__(self, population_size=10, max_gen=10):
        self.train_model_num = 0
        self.population_size = population_size
        self.max_gen = max_gen

    def create_base_family(self, father=None, mother=None):
        seed_father = self.create_seed_net(0, load_network=father)
        seed_mother = self.create_seed_net(1, load_network=mother)
        seed_family = family(self.family_num, seed_father, seed_mother, R=1)
        seed_family.child_num = 1
        self.queue_family.put(seed_family)

    def create_seed_net(self, num, net_name=None, load_network=None):
        seed_net = nn.NeuralNet(0, num, net_name=net_name, load_network=load_network, create_seed=True, is_seed=True)
        seed_net.kick_simulation()
        seed_net.output_deploy_network()
        return seed_net

    def output(self, message):
        print(message)

    def run_family(self):
        if self.queue_family.empty():
            print('have no family!!!!')
            return False
        now_family = self.queue_family.get()
        print('--------------------Family' + str(now_family.family_num) + '--------------------')
        family_h = 'Family ' + str(now_family.family_num)
        family_f = family_h + ' father : net ' + now_family.father.get_fnum_gen()
        family_m = family_h + ' mother : net ' + now_family.mother.get_fnum_gen()
        family_q = family_h + ' Q : ' + str(now_family.Q)
        family_r = family_h + ' R : ' + str(now_family.R)
        self.output(family_f)
        self.output(family_m)
        self.output(family_q)
        self.output(family_r)
        list_new_childs = now_family.create_new_childs()
        self.output('list of good childs:')
        for child in list_new_childs:
            child_gen_q = child.get_fnum_gen() + ':' + str(child.q_value)
            self.output(child_gen_q)
            self.queue_new_child.put(child)
        return True

    def create_new_family(self):
        print('the num of good child list: ' + str(self.queue_new_child.qsize()))
        print('the num of family: ' + str(self.queue_family.qsize()))
        while (self.queue_new_child.qsize() >= 2):
            father = self.queue_new_child.get()
            mother = self.queue_new_child.get()
            self.family_num += 1
            new_family = family(self.family_num, father, mother)
            self.queue_family.put(new_family)

    def check_termination(self):
        if self.family_num < self.max_gen:
            return True
        else:
            return False
