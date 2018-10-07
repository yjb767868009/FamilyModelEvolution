import argparse
import neuroevolution as ne
import os

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('-pz', '--population_size', default=10, type=int, help='population size')
parser.add_argument('-mg', '--max_gen', default=30, type=int, help='max gen')
parser.add_argument('-f', '--first_run', default=0, type=int, help='if you run new genetic')


def main():
    global args
    args = parser.parse_args()
    os.system('rm -r population/*')
    neuro_evo = ne.NeuroEvolution(population_size=args.population_size, max_gen=args.max_gen)
    neuro_evo.create_base_family(father='48',mother='base')
    while (neuro_evo.check_termination()):
        if not neuro_evo.run_family():
            break
        neuro_evo.create_new_family()



if __name__ == '__main__':
    main()