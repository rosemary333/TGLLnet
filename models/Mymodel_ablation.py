# This is the network script
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
import torch
import torch.nn.functional as F
from torch import nn
from models.MRMnet import CNN

class GraphConvolution(Module):
    """
    LGG-specific GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        #self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight)-self.bias
        output = F.relu(torch.matmul(adj, output))
        return output


class GCN(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data):
        graph, adj = data
        adj = self.norm_adj(adj)
        support = torch.matmul(graph, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = (F.relu(output + self.bias), adj)
        else:
            output = (F.relu(output), adj)
        return output

    def norm_adj(self, adj):
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj


class ChebyNet(Module):
    def __init__(self, K, in_feature, out_feature):
        super(ChebyNet, self).__init__()
        self.K = K
        self.filter_weight, self.filter_bias = self.init_fliter(K, in_feature, out_feature)

    def init_fliter(self, K, feature, out, bias=True):
        weight = nn.Parameter(torch.FloatTensor(K, 1, feature, out), requires_grad=True)
        nn.init.normal_(weight, 0, 0.1)
        bias_ = None
        if bias == True:
            bias_ = nn.Parameter(torch.zeros((1, 1, out), dtype=torch.float32), requires_grad=True)
            nn.init.normal_(bias_, 0, 0.1)
        return weight, bias_

    def get_L(self, adj):
        degree = torch.sum(adj, dim=1)
        degree_norm = torch.div(1.0, torch.sqrt(degree) + 1.0e-5)
        degree_matrix = torch.diag(degree_norm)
        # we approximate lambda_max ~= 2
        L = - torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)
        return L

    def rescale_L(self, L):
        largest_eigval, _ = torch.linalg.eigh(L)
        largest_eigval = torch.max(largest_eigval)
        L = (2. / largest_eigval) * L - torch.eye(L.size(0), device=L.device, dtype=torch.float)
        return L

    def chebyshev(self, x, L):
        # to do graph convolution here] X_0 = X, X_1 = L.X, X_k = 2.L.X_(k-1) - X_(k-2)
        x1 = torch.matmul(L, x)
        x_ = torch.stack((x, x1), dim=1)  # (b, 2, chan, fin)
        if self.K > 1:
            for k in range(2, self.K):
                x_current = 2 * torch.matmul(L, x_[:, -1]) - x_[:, -2]  # X_k = 2.L.X_(k-1) - X_(k-2)
                x_current = x_current.unsqueeze(dim=1)
                x_ = torch.cat((x_, x_current), dim=1)

        x_ = x_.permute(1, 0, 2, 3)  # (k, b, chan, fin)   w: (k, 1, fin, fout) f:
        out = torch.matmul(x_, self.filter_weight)  # (k, b, chan, fout)
        out = torch.sum(out, dim=0)  # (b, chan, fout)
        out = F.relu(out + self.filter_bias)
        return out

    def forward(self, data):
        # x: (b, chan, f) adj
        x, adj = data
        L = self.get_L(adj)
        out = self.chebyshev(x, L)
        out = (out, adj)
        return out


class GraphEncoder(nn.Module):
    def __init__(self, num_layers, num_node, in_features, out_features, K,
                 graph2token='Linear', encoder_type='GCN'):
        super(GraphEncoder, self).__init__()
        self.graph2token = graph2token
        self.K = K  # useful for ChebyNet
        assert graph2token in ['Linear', 'AvgPool', 'MaxPool', 'Flatten'], "graph2vector type is not supported!"
        if graph2token == 'Linear':
            self.tokenizer = nn.Linear(num_node*out_features, out_features)
        else:
            self.tokenizer = None
        layers = []
        for i in range(num_layers):
            if i == 0:
                layer = self.get_layer(encoder_type, in_features, out_features)
            else:
                layer = self.get_layer(encoder_type, out_features, out_features)
            layers.append(layer)
        self.encoder = nn.Sequential(*layers)

    def get_layer(self, encoder_type, in_features, out_features):
        assert encoder_type in ['Cheby', 'GCN'], "encoder type is not supported!"
        if encoder_type == 'GCN':
            GNN = GCN(in_features, out_features)
        if encoder_type == 'Cheby':
            GNN = ChebyNet(self.K, in_features, out_features)
        return GNN

    def forward(self, x, adj):
        # x: b, channel, feature
        # adj: m, n, n
        output = self.encoder((x, adj))
        x, _ = output
        if self.tokenizer is not None:
            x = x.view(x.size(0), -1)
            output = self.tokenizer(x)
        else:
            if self.graph2token == 'AvgPool':
                output = torch.mean(x, dim=-1)
            elif self.graph2token == 'MaxPool':
                output = torch.max(x, dim=-1)[0]
            else:
                output = x.view(x.size(0), -1)
        return output


class staticGCN(nn.Module):
    def __init__(self, layers_graph=[1, 2], layers_transformer=1, num_adj=3, num_chan=62,
                 num_feature=5, hidden_graph=16, K=2, num_head=8, dim_head=16,
                 dropout=0.25,  alpha=0.25, graph2token='Linear', encoder_type='GCN', option="GE1", num_classes=10): #GE1,Linear,ST-ReGE, 
        super(staticGCN, self).__init__()
        self.graph_encoder_type = encoder_type
        self.option = option    
        if self.option == 'ST-GeRE' or self.option == 'ECGNN':
            self.GE1 = GraphEncoder(
                num_layers=layers_graph[0], num_node=num_chan, in_features=num_feature,
                out_features=hidden_graph, K=K, graph2token=graph2token, encoder_type=encoder_type
            )
            self.to_GNN_out = nn.Linear(num_chan*num_feature, hidden_graph, bias=False)
            self.adjs = nn.Parameter(torch.FloatTensor(num_adj, num_chan, num_chan), requires_grad=False)
            self.mrm1dd = CNN(input_channels=hidden_graph, mid_channels=16, mu2=1, sigma2=0.1, num_classes=num_classes)

        if graph2token in ['AvgPool', 'MaxPool']:
            hidden_graph = num_chan
        if graph2token == 'Flatten':
            hidden_graph = num_chan*hidden_graph

        if self.option =='wo1dcnn':
            self.fc =  nn.Linear(hidden_graph, num_classes)
        if self.option=='woRMPG':
            self.mrm2dd = CNN(input_channels=12, mid_channels=16, mu2=1, sigma2=0.1,num_classes=num_classes)
    def forward(self, x):
        # x: batch, sequence, chan

        if self.option == 'ST-GeRE' or self.option == 'ECGNN' or self.option=='woTCC':
            x = x.transpose(1, 2)
            b,s,chan = x.size()
            x = x.reshape(b,s//1, chan, 1)
            b, s, chan, f = x.size()
            x = rearrange(x, 'b s c f  -> (b s) c f')

            if self.graph_encoder_type == 'Cheby':
                adjs = self.get_adj(self_loop=False)
            else:
                adjs = self.get_adj()
        
            # multi-view pyramid residual GNN block
            x_ = x.view(x.size(0), -1)
            x_ = self.to_GNN_out(x_) #(b*s,c,f)  (32*100,12,10)
            
            x1 = self.GE1(x, adjs[0])  #(32*100,12,1)
            # x2 = self.GE2(x, adjs[0])  #(32*1)
            x = torch.stack((x1, x_), dim=1)
            x = torch.mean(x, dim=1)

            # temporal contextual transformer
            x = rearrange(x, '(b s) h -> b s h', b=b, s=s)
            if self.option=='woTCC':
                x = torch.mean(x,dim=1).squzze()
                x = self.fc(x)
            else:
                x = self.mrm1dd(x.transpose(1, 2))
            return x,adjs[0]
        elif self.option == 'woRMPG':
            x = self.mrm2dd(x)
            return x,0
            
    def get_adj(self, self_loop=True):
        ######## graph from ST-ReGE
        if self.option=='ST-GeRE':
            adj = torch.FloatTensor(   
               [[0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
                [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]]
            ).to(self.adjs.device)
            adj = adj.unsqueeze(0).repeat(self.adjs.shape[0], 1, 1)

        ####### graph from ECGNN
        if self.option=='ECGNN': 
            adj = torch.FloatTensor( 
               [[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            ).to(self.adjs.device)
            adj = adj.unsqueeze(0).repeat(self.adjs.shape[0], 1, 1)
        return adj



def Mymodel_ablation(**kwargs):
    model = staticGCN(layers_graph=[4], layers_transformer=4, num_adj=1,
              num_chan=12, num_feature=1, hidden_graph=32,
              K=3, num_head=8, dim_head=32, dropout=0.25,
              graph2token='Linear', encoder_type='GCN', alpha=0.25, option='ST-GeRE', **kwargs) #'Linear' 'ST-GeRE' 'ECGNN','woRMPG','woTCC'
    return model



