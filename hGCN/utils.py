import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
import pandas as pd
import math
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import random
import torch.nn as nn

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

def plot_confusion_matrix(y_true, output, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, filename = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    y_pred = output.max(1)[1].type_as(y_true)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes_val = list(classes.keys())
    classes_val = unique_labels(classes_val)[unique_labels(y_true, y_pred)]
    classes_name = [classes[i] for i in classes_val]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes_name, yticklabels=classes_name,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=6.5)
    fig.tight_layout()
    if filename:
        np.save(filename + '.npy', cm)
        fig.savefig(filename + '.png')
    return ax

def top_k_accuracy(output, labels, k):
    pred = torch.topk(output, k, dim=1)[1].type_as(labels)
    correct = pred.eq(labels.view(-1, 1).expand_as(pred)).double()
    correct = correct.sum()
    return correct / len(labels)

def calculate_laplacian(adj):
    total_nodes = adj.shape[0]
    adj = adj + sp.identity(total_nodes)
    degree = adj.sum(axis=1)
    degree_sqrt_inv = np.sqrt(1.0 / degree).reshape(1, -1)
    D_sqrt_inv = sp.diags(degree_sqrt_inv, [0], shape=(total_nodes, total_nodes))
    adj_tilde = np.dot(np.dot(D_sqrt_inv, adj), D_sqrt_inv)
    return adj_tilde

def calculate_l2(index, pair_list, output, labels, CUDA=True):
    same_label_loss = 0
    same_label_count = 0
    diff_label_loss = 0
    diff_label_count = 0
   
    for i in pair_list:
        if i == index:
            continue
        if labels[i] == labels[index]:
            same_label_loss += torch.dist(output[i], output[index], 2)
            same_label_count += 1
        else:
            diff_label_loss += torch.dist(output[i], output[index], 2)
            diff_label_count += 1
    if same_label_count == 0:
        if CUDA:
            return torch.Tensor([0]).cuda(0), diff_label_loss / diff_label_count
        else:
            return torch.Tensor([0]), diff_label_loss / diff_label_count
    elif diff_label_count == 0:
        if CUDA:
            return same_label_loss / same_label_count, torch.zero().cuda(0)
        else:
            return same_label_loss / same_label_count, torch.Tensor([0])
    else:
        return same_label_loss / same_label_count, diff_label_loss / diff_label_count
        
def hybrid_loss(output, encoding, target):
    cnl_f = nn.CrossEntropyLoss()
    cnl = cnl_f(output, target)
    same_label_loss = 0
    diff_label_loss = 0
    random.seed(11)
    left_samples = random.sample(range(encoding.shape[0]), math.ceil(encoding.shape[0] / 500))
    random.seed(28)
    right_sample = random.sample(range(encoding.shape[0]), math.ceil(encoding.shape[0] / 500))
    for i in left_samples:
        sl, dl = calculate_l2(i, right_sample, encoding, target)
        same_label_loss += sl
        diff_label_loss += dl
  
    return 0.8 * cnl + 0.2 * ((sl - dl)) ** 2

def multi_relation_load(path='../data', label="full_labels.csv", 
                        files=["full_adj_phone.npz", "full_adj_app_installed.npz", "full_adj_app_active.npz"],
                        label_mapping="label_mapping",
                        train='full_train_idx', test='full_test_idx'):

    print("Loading data from path {0}".format(path))
    DATA = Path(path)
    ADJS = [DATA/i for i in files]
    LABEL = DATA/label
    adjs = []
    label_df = pd.read_csv(LABEL)
    for adj in ADJS:
        adj_tilde = calculate_laplacian(sp.load_npz(adj))
        adjs.append(adj_tilde)
    
    with open(DATA/train, 'rb') as f:
        idx_train = pickle.load(f)
    with open(DATA/test, 'rb') as f:
        idx_test = pickle.load(f)
    f.close()
    
    with open(DATA/label_mapping, 'rb') as f:
        label_mapping = pickle.load(f)
    
    labels = label_df['group'].values
    
    total_node = adjs[0].shape[0]
    edge_indexs = np.array(range(total_node))

    self_loop = sp.csr_matrix((np.ones(total_node), (edge_indexs, edge_indexs)), 
                              shape=(total_node, total_node), dtype=np.float32)
    adjs.append(self_loop)

    num_neighbors = [adjs[i].sum(1).reshape(total_node, 1) for i in range(len(adjs))]    
    for neighbor in num_neighbors:
        neighbor += 1 # smoothing
    num_neighbors = [torch.Tensor(neighbors) for neighbors in num_neighbors]

    print("\tprocessing features")
    
    features = sp.csr_matrix((np.ones(total_node), (edge_indexs, edge_indexs)), shape=(total_node, total_node), dtype=np.float32) # dummy feature
    
    print("\ttransfering into tensors")

    features = sparse_mx_to_torch_sparse_tensor(features)
    labels = torch.LongTensor(labels)
    adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    num_relation = len(files)
    return adjs, features, labels, label_mapping, idx_train, idx_test, num_neighbors, num_relation
