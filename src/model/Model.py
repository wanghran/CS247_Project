from model.layer import GraphConvolution
import torch.nn as nn
import torch.nn.functional as F

class rGCN(nn.Module):
    def __init__(self, num_relation, num_neighbors, nfeat, nhid, nclass, dropout):
        super(rGCN, self).__init__()

        self.gc1 = GraphConvolution(num_relation, num_neighbors, nfeat, nhid)
        self.gc2 = GraphConvolution(num_relation, num_neighbors, nhid, nclass)
        self.dropout = dropout

        
    '''
    featureless forward function. First x input is a (n x n) all one matrix.
    '''
    def forward(self, x, adjs):
        x = F.relu(self.gc1(x, adjs))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adjs)
        return F.log_softmax(x, dim=1)
