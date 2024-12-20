"""
Pre compute laplacians and gradient features of the model in ShapeNet
Because those operators are computed on cpu and can be time consuming, it's possible to process only a chunk of the dataset, to spread the workload across several cpus
"""
from shapenet_dataset import ShapenetDataset
import argparse
import random
import numpy as np
from tqdm import tqdm

random.seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--low', type=int, help='Lower bound index of the chunk of the dataset tyo process')
parser.add_argument('--high', type=int, help='Upper bound index of the chunk of the dataset tyo process')
parser.add_argument('--k_eig', type=int, default=128, help='Size of eigen decomposition')
parser.add_argument('--split', type=str, default='train', help='split to process, choice between val and train')
args = parser.parse_args()

dataset = ShapenetDataset(k_eig=args.k_eig, split='val')
print(f'Processing {args.high-args.low + 1} samples')

# Automatically computes the operators during instanciation if they were not before
for i in tqdm(range(args.low, args.high)):
    sample = dataset[i]
