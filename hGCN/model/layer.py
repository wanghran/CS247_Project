import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple rGCN layer, similar to https://arxiv.org/abs/1703.06103
    :param num_relation: number of different relations in the data
    :param num_neighbors: a #relation x #node x 1 matrix that denotes number of neighbors of a node in a relation
    :param in_features: number of feature of the input
    :param out_features: number of feature of the ouput
    :param bias: if bias is added, default is True
    :type num_relation: int
    :type num_relation: int
    :type num_neighbors: array-like object, must be 3 dimension
    :type in_features: int
    :type out_features: int
    :type bias: bool
    """

    def __init__(self, num_relation, num_neighbors, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_weight = Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(num_relation + 1, in_features, out_features)))
        self.num_neighbors = num_neighbors
        self.attention = Parameter(nn.init.uniform_(torch.FloatTensor(num_relation + 1)))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.adj_weight.size(2))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adjs):
        outputs = []
        num_node = adjs[0].shape[0]
        for i in range(len(self.adj_weight)):
            support = torch.mm(input, self.adj_weight[i])
            output = torch.spmm(adjs[i], support)
            output = output * F.softmax(self.attention, dim=0)[i]
            outputs.append(output)
        output = sum(outputs)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'