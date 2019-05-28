import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
import pandas as pd
import math

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

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

def multi_relation_load(path="../../data/twitter_1hop", label="dict.csv", 
                        files=["friend_list.csv", "retweet_list.csv"]):
    print("Loading data from path {0}".format(path))
    DATA = Path(path)
    FILE = [DATA/i for i in files]
    LABEL = DATA/label
    data_dfs = []
    label_df = pd.read_csv(LABEL, sep="\t")
    for file in FILE:
        data_dfs.append(pd.read_csv(file, sep="\t"))
      
    labeled_ids = label_df["twitter_id"].values
    labels = [0 if v == "D" else 1 for v in label_df["party"].values]
    n_labels = len(labels)
    n_train = math.ceil(n_labels / 10. * 3)
    n_valid = math.ceil(n_labels / 10. * 3)
    idx_train = range(n_train)
    idx_val = range(n_train, n_train + n_valid)
    idx_test = range(n_train + n_valid, n_labels)
    
    print("\tprocessing nodes")
    
    all_ids = set()
    for df in data_dfs:
        all_ids = all_ids.union(set(df[df.columns[0]]))
        all_ids = all_ids.union(set(df[df.columns[1]]))
    unlabeled_ids = all_ids - set(labeled_ids)
    all_id_list = np.concatenate((np.array(labeled_ids, dtype=np.int64), 
                                 np.array(list(unlabeled_ids), dtype=np.int64)))
    n_entities = len(all_id_list)
    idx_map = {j: i for i, j in enumerate(all_id_list)}
    
    print("\tprocessing edges")

    adjs = []
    
    for data_df in data_dfs:
        from_idx = np.array(list(map(idx_map.get, data_df[data_df.columns[0]].values)), dtype=np.int64)
        to_idx = np.array(list(map(idx_map.get, data_df[data_df.columns[1]].values)), dtype=np.int64)
        counts = np.array(data_df[data_df.columns[2]].values, dtype=np.int64)

        n_edges = len(from_idx)

        adj = sp.csr_matrix((counts, (from_idx, to_idx)),
                        shape=(n_entities, n_entities),
                        dtype=np.float32)

        # if build symmetric adjacency matrix
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = normalize(adj + np.eye(adj.shape[0]))
        # adjs.append(adj)
        # otherwise
        adjs.append(adj)
        adjs.append(adj.T)
    
    edge_indexs = np.array(range(n_entities))
    self_loop = sp.csr_matrix((np.ones(n_entities), (edge_indexs, edge_indexs)), 
                              shape=(n_entities, n_entities), dtype=np.float32)
    adjs.append(self_loop)

    num_neighbors = [adjs[i].sum(1).reshape(n_entities, 1) for i in range(len(adjs))]    
    # num_neighbors = [np.diff(adjs[i].indptr).reshape(n_entities, 1) for i in range(len(adjs))]
    for neighbor in num_neighbors:
        neighbor += 1 # smoothing
    num_neighbors = [torch.Tensor(neighbors) for neighbors in num_neighbors]

    print("\tprocessing features")
    
    # edge_indexs = np.array(range(n_entities))
    # features = sp.csr_matrix((np.ones(n_entities), (edge_indexs, edge_indexs)), shape=(n_entities, n_entities), dtype=np.float32) # dummy feature
    features = np.ones((n_entities, 20))
    
    print("\ttransfering into tensors")
    
    features = normalize(features)
    # features = sparse_mx_to_torch_sparse_tensor(features)
    features = torch.Tensor(features)
    labels = torch.LongTensor(labels)
    adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adjs, features, labels, idx_train, idx_val, idx_test, num_neighbors
