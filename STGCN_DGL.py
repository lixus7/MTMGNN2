import sys
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from torchsummary import summary
from Param import *

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
    def __init__(self, dropout=0.3, in_dim=2, out_dim=12,embed_size=64,kernel_size=2, blocks=1, layers=1):
        super().__init__()
        print("========batch_g_gat_2l===========")
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

    def forward(self, x, g):
        # Input shape is (bs, features, n_nodes, n_timesteps)  [B,C,N,T]
#         print("===0 x.shape: ", x.shape)  # torch.Size([batch, 2, 81, 12]) 
        x1 = self.start_conv(x)
#         print("1 x1:", x1.shape)
        x2 = F.leaky_relu(self.cat_feature_conv(x))
#         print("2 x2:", x2.shape)
        x = x1 + x2
#         print("3 x:", x.shape)
        skip = 0

        # STGAT layers
        [batch_size, fea_size, num_of_vertices, step_size] = x.size()
        
        batched_g = dgl.batch(batch_size * [g])

        h = x.permute(0, 2, 1, 3).reshape(batch_size*num_of_vertices, fea_size*step_size)
#         print("6 h:", h.shape)

        h = self.gat_layers1(batched_g, h).mean(1)
#         print("7 h:", h.shape)
        h = self.gat_layers2(batched_g, h).mean(1)
#         print("8 h:", h.shape)
        gc = h.reshape(batch_size, num_of_vertices, fea_size, -1)
#         print("9 gc:", gc.shape)
        graph_out = gc.permute(0, 2, 1, 3)
#         print("10 graph_out:", graph_out.shape)
        output = x + graph_out
#         print("11 output:", output.shape)
#         x = F.relu(x)  # ignore last X
#         print("return output:", output.shape)   # [32, 64, 81, 12]
        return output


'''
align主要是对数据格式进行一个处理，类似于reshape
'''

class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x
'''
门控卷积单元的定义，目的是用来提取时空特征
GLU和sigmoid
'''
class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(temporal_conv_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)
'''
空间卷积层
'''

# class spatio_conv_layer(nn.Module):
#     def __init__(self, ks, c, Lk):
#         super(spatio_conv_layer, self).__init__()
#         self.Lk = Lk
#         self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
#         self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
#         self.reset_parameters()

#     def reset_parameters(self):
#         init.kaiming_uniform_(self.theta, a=math.sqrt(5))
#         fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
#         bound = 1 / math.sqrt(fan_in)
#         init.uniform_(self.b, -bound, bound)

#     def forward(self, x):
#         x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)
#         x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
#         return torch.relu(x_gc + x)


class spatio_conv_layer1(nn.Module):
    def __init__(self, ks, c, G, device):
        super(spatio_conv_layer1, self).__init__()
        self.G = G
#         self.gatlayer = GAT(device = device, in_channels = c, embed_dim = c , aggregate='average')
        self.gatlayer = GAT(dropout=0.3, in_dim=c, out_dim=10,embed_size=c, kernel_size=2, blocks=1, layers=1)   #demand input [B,C,N,T]
        
    def forward(self, x):   # x [B,C,T,N]
#         print('x shape: ',x.shape)
        gat_input = x.permute(0,1,3,2)
        gat_out = self.gatlayer(gat_input,self.G)   # [B,C,N,T]
        gat = gat_out.permute(0,1,3,2)

        return gat

class spatio_conv_layer2(nn.Module):
    def __init__(self, ks, c, G, device):
        super(spatio_conv_layer2, self).__init__()
        self.G = G
#         self.gatlayer = GAT(device = device, in_channels = c, embed_dim = c , aggregate='average')
        self.gatlayer = GAT(dropout=0.3, in_dim=c, out_dim=6,embed_size=c, kernel_size=2, blocks=1, layers=1)   #demand input [B,C,N,T]
        
    def forward(self, x):   # x [B,C,T,N]
#         print('x shape: ',x.shape)
        gat_input = x.permute(0,1,3,2)
        gat_out = self.gatlayer(gat_input,self.G)   # [B,C,N,T]
        gat = gat_out.permute(0,1,3,2)

        return gat
    
'''
这就是对应的一个ST_conv模块，会调用到前面定义好的tconv和sconv
'''
class st_conv_block1(nn.Module):
    def __init__(self, ks, kt, n, c, p, G, device):
        super(st_conv_block1, self).__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
        self.sconv = spatio_conv_layer1(ks, c[1], G, device)
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])

        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)        
        return self.dropout(x_ln)

class st_conv_block2(nn.Module):
    def __init__(self, ks, kt, n, c, p, G, device):
        super(st_conv_block2, self).__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
        self.sconv = spatio_conv_layer2(ks, c[1], G, device)
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])

        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)        
        return self.dropout(x_ln)    


class output_layer(nn.Module):
    def __init__(self, c, T, n):
        super(output_layer, self).__init__()
        self.tconv1 = temporal_conv_layer(T, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        self.fc = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)

class STGCN_DGL(nn.Module):
    def __init__(self, ks, kt, bs, T, n, G, p, device):
        super(STGCN_DGL, self).__init__()
        self.st_conv1 = st_conv_block1(ks, kt, n, bs[0], p, G, device)
        self.st_conv2 = st_conv_block2(ks, kt, n, bs[1], p, G, device)
        self.output = output_layer(bs[1][2], T - 4 * (kt - 1), n)

    def forward(self, x):
        x_st1 = self.st_conv1(x)
        x_st2 = self.st_conv2(x_st1)
#         return torch.relu(self.output(x_st2))
        return self.output(x_st2)
    
    
def load_matrix(file_path):
    adj_mx = pd.read_csv(file_path)
#     adj_mx = adj_mx.replace(0,np.inf)
    return adj_mx

'''
exp( dij,thema)
'''
def weight_matrix(W, sigma2=0.1, epsilon=0.5, scaling=True):
    '''“”
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W/10000
        W2 = W * W
        W_mask = (np.ones([n, n]) - np.identity(n))
        W_mask2 = W_mask*(np.exp(-W2) < 1)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W


'''
lap变换
'''

def scaled_laplacian(W):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    '''
    # d ->  diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))
'''
切比雪夫多项式
'''
#aguin
def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def print_params(model_name, model):
    param_count=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += param.numel()
    print(f'{model_name}, {param_count} trainable parameters in total.')
    return 
# from tensorboardX import SummaryWriter    
def main():
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    
    # ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 32, 64], [64, 32, 128]], TIMESTEP_IN, N_NODE, 0
    ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], INPUT_STEP, N_NODE, 0
    A = pd.read_csv(ADJPATH).values
    G = get_normalized_adj(A)
    W = weight_matrix(A)
    L = scaled_laplacian(W)
    Lk = cheb_poly(L, ks)
    Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
#     print(W)
#     G = torch.Tensor(G.astype(np.float32)).to(device)
    
    g = sp.csr_matrix(G)
    ## 方式2: 使用稀疏矩阵进行构造
    g2 = dgl.from_scipy(g)
    g2 = g2.to(device)
    
    X = torch.Tensor(torch.randn(8,CHANNEL,12,81)).to(device)
    model = STGCN_DGL(ks, kt, bs, T, n, g2, p, device).to(device)
    model(X)
#     summary(model, (CHANNEL, INPUT_STEP, N_NODE), device=device)
    
    print_params('STGCN_GAT1_TB',model)
#     writer = SummaryWriter(log_dir='logs')
#     # 将网络net的结构写到logs里：
#     data = torch.rand(8,2,12,81).to(device)
#     writer.add_graph(model,input_to_model=(data,))
    
if __name__ == '__main__':
    main()  
