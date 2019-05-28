import numpy as np
import torch
import math
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse

from utils import multi_relation_load
from model.Model import rGCN
from train import train
from test import test
# import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data', type=str, default='../../data/twitter_1hop',
                    help='path of the data folder.')
parser.add_argument('--relations', type=str, default=['friend_list.csv'], action='append',
                    help='edge list files for different relations')

args = parser.parse_args()

CUDA = torch.cuda.is_available()
np.random.seed(42)
torch.manual_seed(42)
if CUDA:
    torch.cuda.manual_seed(42)

DATA = Path(args.data)

adjs, features, labels, idx_train, idx_val, idx_test, num_neighbors = multi_relation_load(DATA, files=args.relations)

if CUDA:
    features = features.cuda(0)
    adjs = [i.cuda(0) for i in adjs]
    labels = labels.cuda(0)
    idx_train = idx_train.cuda(0)
    idx_val = idx_val.cuda(0)
    idx_test = idx_test.cuda(0)
    num_neighbors = [neighbors.cuda(0) for neighbors in num_neighbors]

rGCN_model = rGCN(len(args.relations),
            num_neighbors,
            features.shape[1],
            args.hidden,
            labels.max().item() + 1,
            args.dropout)
optimizer = optim.Adam(rGCN_model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


if CUDA:
    # rGCN_model = nn.DataParallel(rGCN_model)
    rGCN_model.cuda(0)

# t_total = time.time()
print("start training")
train([features, adjs, idx_train, idx_val, labels], rGCN_model, optimizer, args.epochs)
# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
test([features, adjs, idx_test, labels], rGCN_model)
