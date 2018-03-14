import argparse
import torch
import torch.nn as nn
from network import SRCNN
parser = argparse.ArgumentParser()

parser.add_argument('--dataRoot', required=False, help='path to dataset')
parser.add_argument('--ngpu', type=int, default=2, help='number of ngpu using')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')

args = parser.parse_args()

model = SRCNN(3,N_gpu=args.ngpu)