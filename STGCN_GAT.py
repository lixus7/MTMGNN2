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
ref Diego999
https://github.com/Diego999/pyGAT
'''

import torch
from torch import nn
import sys
import scipy.sparse as sp
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
class GraphAttention(nn.Module):
    def __init__(self, device,in_channels, embed_dim, dropout=0.1, alpha=0.2, bias=True):
        super(GraphAttention, self).__init__()
        self.embed_dim = embed_dim
        self.W = nn.Parameter(torch.empty(size=(in_channels, embed_dim)))
#         self.W = nn.Parameter(torch.empty([embed_dim, in_channels]), requires_grad=True)
        self.a = nn.Parameter(torch.empty(2*embed_dim,1), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(in_channels), requires_grad=True) if bias else None
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.device  = device
        self.Tanh = nn.Tanh()
        
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    def forward(self, x: torch.Tensor, adj: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): shape in [batch, nodes, in_channels]
            mask (torch.Tensor, optional): shape is [batch, nodes, nodes]. Defaults to None.
        Returns:
            [torch.Tensor]: shape is [batch, nodes, embed_dim]
        """
#         print('cuda device is :',self.device)
#         print('0 input x shape :',x.shape) # [B, N, C]
        hidden = torch.matmul(x,self.W)
#         print('1 hidden shape :',hidden.shape)  # [B, N, embed_dim]
        B = hidden.shape[0]
        N = hidden.shape[1]
        hidden_repeated_in_chunks = hidden.repeat_interleave(N, dim=1)
        hidden_repeated_alternating = hidden.repeat(1, N, 1)
#         print(' hidden_repeated_alternating shape :',hidden_repeated_alternating.shape) 
        all_combinations_matrix = torch.cat([hidden_repeated_in_chunks, hidden_repeated_alternating], dim=1).view(B, N, N, 2 * self.embed_dim)
#         print(' all_combinations_matrix shape :',all_combinations_matrix.shape) 
        attn= torch.matmul(all_combinations_matrix,self.a.to(self.device)).squeeze(-1) # [B, N, N]  
#         print('2 attn shape:',attn.shape)      
        # leaky ReLU
        attn = self.leaky_relu(attn)   # [B, N, N]  
#         print('3 leaky ReLU attn shape :',attn.shape)
        # adj
        zero_vec = -9e15*torch.ones_like(attn)
        attn = torch.where(adj > 0, attn, zero_vec)
#         if adj is not None:
#             attn += torch.where(adj > 0.0, 0.0, -1e12)
#         # add bias
#         if self.bias is not None:
#             out += self.bias.to(self.device)
        # softmax
        attention = F.softmax(attn, dim=-1)  # [B, N, N]
#         print('4 attention softmax: ',attention.shape)
        # dropout
        attention = F.dropout(attention, self.dropout, training=self.training)
        out = torch.matmul(attention, x)  #   atten [B,N,N] *  hidden [B,N,embed_dim = 64]     ---->  output [B, N, embed_dim]     feature * att_score
        out = self.Tanh(out)
        return out  # [B, N, embed_dim]
        

class GAT(nn.Module):
    """
    Graph Attention Network
    """

    def __init__(self, device ,in_channels, embed_dim,
                 n_heads=8, dropout=0.1, alpha=0.2, bias=True,aggregate='average'):
        """
        Args:
            aggregate='concat'
            in_channels ([type]): input channels
            embed_dim ([type]): output channels
            n_heads (int, optional): number of heads. Defaults to 64.
            dropout (float, optional): dropout rate. Defaults to .1.
            alpha (float, optional): leaky ReLU negative_slope. Defaults to .2.
            bias (bool, optional): use bias. Defaults to True.
            aggregate (str, optional): aggregation method. Defaults to 'concat'.
        """
        super(GAT, self).__init__()
        assert aggregate in ['concat', 'average']
        self.attns = nn.ModuleList([
            GraphAttention(device , in_channels, embed_dim, dropout=dropout, alpha=alpha, bias=bias)
            for _ in range(n_heads)
        ])
        self.dropout = dropout
        self.aggregate = aggregate

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): shape is [batch, nodes, in_channels]
            adj (torch.Tensor, optional): shape is [batch, nodes, nodes]. Defaults to None.
        """
        x = F.dropout(x, self.dropout, training=self.training)
        if self.aggregate == 'concat':
            output = torch.cat([attn(x, adj) for attn in self.attns], dim=-1)
#             print('output shape:',output.shape)    #  [2, 81, 512]
            output = F.dropout(output, self.dropout, training=self.training)   #  [2, 81, 64]
            return F.elu(output)
        else:
            output = torch.mean(torch.stack([attn(x, adj) for attn in self.attns]),dim=0)  #  [2, 81, 64]
#             print('output shape:',output.shape)
#             output = sum([attn(x, adj) for attn in self.attns]) / len(self.attns)
            output = F.dropout(output, self.dropout, training=self.training)   #  [2, 81, 64]
#         print('return out :',F.elu(output).shape)
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

class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c, Lk, device):
        super(spatio_conv_layer, self).__init__()
        self.Lk = Lk
        
        self.gatlayer = GAT(device = device, in_channels = c, embed_dim = c , aggregate='average')

    def forward(self, x):   # x [B,C,T,N]
        T_shape = x.shape[2]
        GAT_lst = list()
        for i in range(T_shape):
            in_GAT = x[:,:,i,:].permute(0,2,1)   # [B,Ci,N]   ---[B,N,Ci] 
            out_GAT = self.gatlayer(in_GAT,self.Lk)
            GAT_lst.append(out_GAT)
        GAT_feature = torch.stack(GAT_lst, dim=2).permute(0,3,2,1) #[B,Co,T,N] Co=bs  
#         print('GAT_feature shape： ',GAT_feature.shape)
        sp_out = x + GAT_feature
        return sp_out

'''
这就是对应的一个ST_conv模块，会调用到前面定义好的tconv和sconv
'''
class st_conv_block(nn.Module):
    def __init__(self, ks, kt, n, c, p, Lk, device):
        super(st_conv_block, self).__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
        self.sconv = spatio_conv_layer(ks, c[1], Lk, device)
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

class STGCN(nn.Module):
    def __init__(self, ks, kt, bs, T, n, Lk, p, device):
        super(STGCN, self).__init__()
        self.st_conv1 = st_conv_block(ks, kt, n, bs[0], p, Lk, device)
        self.st_conv2 = st_conv_block(ks, kt, n, bs[1], p, Lk, device)
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

# from tensorboardX import SummaryWriter    
def main():
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    
    # ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 32, 64], [64, 32, 128]], TIMESTEP_IN, N_NODE, 0
    ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], INPUT_STEP, N_NODE, 0
    A = pd.read_csv(ADJPATH).values
    W = get_normalized_adj(A)
#     L = scaled_laplacian(W)
#     Lk = cheb_poly(L, 1)
#     print(W)
    W = torch.Tensor(W.astype(np.float32)).to(device)
    X = torch.Tensor(torch.randn(8,1,12,81)).to(device)
    model = STGCN(ks, kt, bs, T, n, W, p, device).to(device)
    model(X)
    summary(model, (CHANNEL, INPUT_STEP, N_NODE), device=device)
#     writer = SummaryWriter(log_dir='logs')
#     # 将网络net的结构写到logs里：
#     data = torch.rand(8,2,12,81).to(device)
#     writer.add_graph(model,input_to_model=(data,))
    
if __name__ == '__main__':
    main()  
