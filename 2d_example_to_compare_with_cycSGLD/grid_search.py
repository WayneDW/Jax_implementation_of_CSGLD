#!/usr/bin/env python3
import random
import os

secure_random = random.SystemRandom()

'''
parser.add_argument('-lr',  default=0.02, type=float)
parser.add_argument('-zeta',  default=0.6, type=float)
parser.add_argument('-sz',  default=10, type=float)
parser.add_argument('-temperature',  default=1, type=float)
parser.add_argument('-num_partitions',  default=10000, type=int)
parser.add_argument('-energy_gap',  default=0.001, type=float)
parser.add_argument('-seed',  default=1, type=int)
'''


for _ in range(10):
    seed = random.randint(1, 10**5)
    lr = secure_random.choice([0.02, 0.01, 0.005])
    zeta = secure_random.choice([0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 0.8, 1])
    sz = secure_random.choice([10, 6, 3, 1, 0.6, 0.3, 0.1, 0.06, 0.03, 0.01])
    temperature = 1.
    energy_gap, num_partitions = secure_random.choice([(0.001, 10000), (0.0003, 30000), (0.0001, 100000)])
    os.system(f'python3 howto_use_csgld_2d_example_compare_with_csgld.py -lr {lr} -zeta {zeta} -sz {sz} -temperature {temperature} -energy_gap {energy_gap} -num_partitions {num_partitions} -seed {seed}')
