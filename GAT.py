'''
dgl
'''

import torch.nn.functional as F
import math
from dgl.nn.pytorch import edge_softmax, GATConv
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import torch.nn.functional as F
import dgl
import sys
import scipy.sparse as sp
import numpy as np
from torchsummary import summary

class GAT(nn.Module):
    def __init__(self, g, dropout=0.3, in_dim=2, out_dim=12,embed_size=64,kernel_size=2, blocks=1, layers=1):
        super().__init__()
        print("========batch_g_gat_2l===========")
        self.g = g
        self.dropout = dropout
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,out_channels=embed_size,kernel_size=(1, 1))
        self.cat_feature_conv = nn.Conv2d(in_channels=in_dim,out_channels=embed_size,kernel_size=(1, 1))

        heads = 8
        feat_drop = 0.6
        attn_drop = 0.6
        negative_slope = 0.2

        self.gat_layers1 = GATConv(embed_size*out_dim,embed_size*out_dim,
            heads, feat_drop, attn_drop, negative_slope,residual=False, activation=F.elu)
        self.gat_layers2 = GATConv(embed_size*out_dim,embed_size*out_dim,
            heads, feat_drop, attn_drop, negative_slope,residual=False, activation=F.elu)


        self.end_conv_2 = Conv2d(embed_size, out_dim, (1, 1), bias=True)

    def forward(self, x):
        # Input shape is (bs, features, n_nodes, n_timesteps)
        print("===0 x.shape: ", x.shape)  # torch.Size([batch, 2, 81, 12]) 
        x1 = self.start_conv(x)
        print("1 x1:", x1.shape)
        x2 = F.leaky_relu(self.cat_feature_conv(x))
        print("2 x2:", x2.shape)
        x = x1 + x2
        print("3 x:", x.shape)
        skip = 0

        # STGAT layers
        [batch_size, fea_size, num_of_vertices, step_size] = x.size()
        
        batched_g = dgl.batch(batch_size * [self.g])

        h = x.permute(0, 2, 1, 3).reshape(batch_size*num_of_vertices, fea_size*step_size)
        print("6 h:", h.shape)

        h = self.gat_layers1(batched_g, h).mean(1)
        print("7 h:", h.shape)
        h = self.gat_layers2(batched_g, h).mean(1)
        print("8 h:", h.shape)
        gc = h.reshape(batch_size, num_of_vertices, fea_size, -1)
        print("9 gc:", gc.shape)
        graph_out = gc.permute(0, 2, 1, 3)
        print("10 graph_out:", graph_out.shape)
        x = x + graph_out
        print("11 x:", x.shape)
#         x = F.relu(x)  # ignore last X
        print("return x:", x.shape)
        return x
    
def print_params(model_name, model):
    param_count=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += param.numel()
    print(f'{model_name}, {param_count} trainable parameters in total.')
    return        
def main():

    GPU = sys.argv[-1] if len(sys.argv) == 2 else '7'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    adj = torch.randn(81, 81)
    
    g = sp.csr_matrix(adj)
    ## 方式2: 使用稀疏矩阵进行构造
    g2 = dgl.from_scipy(g)
    g2 = g2.to(device)
    model =  GAT(g=g2).to(device)
    X = torch.randn(32, 2,81,12).to(device)
    model(X)
    print_params('Transformer',model)
#     summary(model, (2, 81, 12), device=device)
    
if __name__ == '__main__':
    main()       