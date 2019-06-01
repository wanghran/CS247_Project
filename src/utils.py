import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
import pandas as pd
import math
import pickle

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def multi_relation_load(path="../data/", label="labels.csv", 
                        files=["adj_phone.npz", "adj_app_installed.npz", "adj_app_active.npz"],
                        train='train_idx', test='test_idx'):
    print("Loading data from path {0}".format(path))
    DATA = Path(path)
    ADJS = [DATA/i for i in files]
    LABEL = DATA/label
    adjs = []
    label_df = pd.read_csv(LABEL)
    for adj in ADJS:
        adjs.append(sp.load_npz(adj))
    
    with open(DATA/train, 'rb') as f:
        idx_train = pickle.load(f)
    with open(DATA/test, 'rb') as f:
        idx_test = pickle.load(DATA/test)

    labels = label_df['group'].values

    total_node = adjs[0].shape[0]
    edge_indexs = np.array(range(total_node))
    self_loop = sp.csr_matrix((np.ones(total_node), (edge_indexs, edge_indexs)), 
                              shape=(total_node, total_node), dtype=np.float32)
    adjs.append(self_loop)

    num_neighbors = [adjs[i].sum(1).reshape(total_node, 1) for i in range(len(adjs))]    
    # num_neighbors = [np.diff(adjs[i].indptr).reshape(n_entities, 1) for i in range(len(adjs))]
    for neighbor in num_neighbors:
        neighbor += 1 # smoothing
    num_neighbors = [torch.Tensor(neighbors) for neighbors in num_neighbors]

    print("\tprocessing features")
    
    # edge_indexs = np.array(range(n_entities))
    features = sp.csr_matrix((np.ones(total_node), (edge_indexs, edge_indexs)), shape=(total_node, total_node), dtype=np.float32) # dummy feature
    
    print("\ttransfering into tensors")

    features = sparse_mx_to_torch_sparse_tensor(features)
    labels = torch.LongTensor(labels)
    adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    return adjs, features, labels, idx_train, idx_test, num_neighbors
