import argparse
import neuroevolution as ne
import os

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('-pz', '--population_size', default=10, type=int, help='population size')
parser.add_argument('-mg', '--max_gen', default=50, type=int, help='max gen')
parser.add_argument('-f', '--first_run', default=0, type=int, help='if you run new genetic')


def main():
    global args
    args = parser.parse_args()
    os.system('rm -r population/*')
    os.system('rm family_log.txt')
    os.system('rm log.txt')
    neuro_evo = ne.NeuroEvolution(population_size=args.population_size,
                                  max_gen=args.max_gen,
                                  num_elites=10)
    neuro_evo.create_base_family(father='base1', mother='base2')
    while neuro_evo.check_termination():
        if not neuro_evo.run_family():
            break
        neuro_evo.create_new_family()


if __name__ == '__main__':
    main()
