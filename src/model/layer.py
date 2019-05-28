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
        self.adj_weight = Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(num_relation * 2 + 1, in_features, out_features)))
        self.num_neighbors = num_neighbors
        self.feature_weight = Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(in_features, out_features)))
        self.attention = Parameter(nn.init.uniform_(torch.FloatTensor(num_relation * 2 + 1)))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.adj_weight.size(2))
        # self.weight.data.uniform_(-stdv, stdv)
        # self.adj_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adjs):
        outputs = []
        num_node = adjs[0].shape[0]
        for i in range(len(self.adj_weight)):
            support = torch.mm(input, self.adj_weight[i])
            output = torch.spmm(adjs[i], support)
            # output = output / self.num_neighbors[i]
            # output = output * F.softmax(self.attention, dim=0)[i]
            outputs.append(output)
        # output = sum(outputs)
        output = self.normalization(outputs)
        # feature_out = torch.mm(input, self.feature_weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def normalization(self, embedding: list) -> torch.Tensor:
        relations = []
        total_sum = 0.0
        for emb in embedding:
            relations.append(torch.sum(emb, dim=0))
        total_sum = sum(relations)
        relations = [relation / total_sum for relation in relations]

        # print(f"relation size {len(relations)}")

        relation_normalization = 0
        outputs = 0
        if embedding[0].device == 'cuda':
            relation_normalization = torch.stack(relations, dim=0).cuda()
            outputs = torch.stack(embedding, dim=0).cuda()
        else:
            relation_normalization = torch.stack(relations, dim=0)
            outputs = torch.stack(embedding, dim=0)
        assert type(relation_normalization) == torch.Tensor
        assert type(outputs) == torch.Tensor

        attention = []
        for i in range(embedding[0].size(1)):
            attention.append(self.attention)
        attention = torch.stack(attention, dim=1)
        attention = F.softmax(attention * relation_normalization, dim=0).reshape(len(embedding), 1, embedding[0].size(1))
        outputs = attention * outputs
        return torch.sum(outputs, dim=0)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
