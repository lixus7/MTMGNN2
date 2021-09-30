import sys
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from Param import *
from torchsummary import summary

'''
dgl GAT module
'''
from dgl.nn.pytorch import edge_softmax, GATConv
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import dgl
import scipy.sparse as sp

class GAT(nn.Module):
    def __init__(self, device, in_dim=16, out_dim=10,embed_size=16):
        super(GAT,self).__init__()
        print("========batch_g_gat_2l===========")
        self.device = device
        self.start_conv = nn.Conv2d(in_channels=in_dim,out_channels=embed_size,kernel_size=(1, 1))
        self.cat_feature_conv = nn.Conv2d(in_channels=in_dim,out_channels=embed_size,kernel_size=(1, 1))

        heads = 8
        feat_drop = 0.6
        attn_drop = 0.6
        negative_slope = 0.2

        self.gat_layers1 = GATConv(embed_size*out_dim,embed_size*out_dim,heads, feat_drop, attn_drop, negative_slope,residual=False,activation=F.elu)
        self.gat_layers2 = GATConv(embed_size*out_dim,embed_size*out_dim,heads, feat_drop, attn_drop, negative_slope,residual=False,activation=F.elu)  #activation=F.elu

    def forward(self, x:torch.Tensor, g):
        '''
        Param x: input feature - torch.Tensor [B,C,N,T]
        Param g: support adj matrices - [N,N]
        '''
#         G = sp.csr_matrix(g.numpy())
#         G = dgl.from_scipy(G)
#         G = G.to(self.device)
        
        temp_gat = []
        for i in range(x.shape[0]):
            G = sp.csr_matrix(g.numpy())
            G = dgl.from_scipy(G)
            G = G.to(self.device)
            temp_gat.append(G)        
        
        x1 = self.start_conv(x)
        x2 = F.leaky_relu(self.cat_feature_conv(x))
        x = x1 + x2
        # GAT layers
        [batch_size, fea_size, num_of_vertices, step_size] = x.size()       
#         batched_g = dgl.batch(batch_size * [G])
        batched_g = dgl.batch(temp_gat)
        h = x.permute(0, 2, 1, 3).reshape(batch_size*num_of_vertices, fea_size*step_size)

        h = self.gat_layers1(batched_g, h).mean(1)
        h = self.gat_layers2(batched_g, h).mean(1)
        gc = h.reshape(batch_size, num_of_vertices, fea_size, -1)
        graph_out = gc.permute(0, 2, 1, 3)
        output = x + graph_out
        return output

    
class GAT_D(nn.Module):
    def __init__(self, device, in_dim=16, out_dim=10,embed_size=16):
        super(GAT_D, self).__init__()
        print("========batch_g_gat_2l===========")
        self.start_conv = nn.Conv2d(in_channels=in_dim,out_channels=embed_size,kernel_size=(1, 1))
        self.cat_feature_conv = nn.Conv2d(in_channels=in_dim,out_channels=embed_size,kernel_size=(1, 1))
        self.device = device
        heads = 8
        feat_drop = 0.6
        attn_drop = 0.6
        negative_slope = 0.2

        self.gat_layers1 = GATConv(embed_size*out_dim,embed_size*out_dim,heads, feat_drop, attn_drop, negative_slope,residual=False,activation=F.elu)
        self.gat_layers2 = GATConv(embed_size*out_dim,embed_size*out_dim,heads, feat_drop, attn_drop, negative_slope,residual=False,activation=F.elu)  #activation=F.elu

    def forward(self, x:torch.Tensor, g):
        '''
        Param x: input feature - torch.Tensor [B,C,N,T]
        Param g: norm_dynamic_adj [B,N,N]
        ''' 
        temp_gat = []
        for i in range(g.shape[0]):
            G = sp.csr_matrix(g[i,:,:].numpy())
            G = dgl.from_scipy(G)
            G = G.to(self.device)
            temp_gat.append(G)
        
        x1 = self.start_conv(x)
        x2 = F.leaky_relu(self.cat_feature_conv(x))
        x = x1 + x2
        # GAT layers
        [batch_size, fea_size, num_of_vertices, step_size] = x.size()
        batched_g = dgl.batch(temp_gat)
        h = x.permute(0, 2, 1, 3).reshape(batch_size*num_of_vertices, fea_size*step_size)

        h = self.gat_layers1(batched_g, h).mean(1)
        h = self.gat_layers2(batched_g, h).mean(1)
        gc = h.reshape(batch_size, num_of_vertices, fea_size, -1)
        graph_out = gc.permute(0, 2, 1, 3)
        output = x + graph_out
        return output
    
'''
padding and reshape
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
TCN module
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
GCN module
'''
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        '''
        Param x: input feature - torch.Tensor [B,C,N,T]
        Param A: support adj matrices - [N,N]
        '''                          
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()    
class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)    
    
class GCN(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=2,order=2):
        super(GCN,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        '''
        Param x: input feature - torch.Tensor [B,C,N,T]
        Param support: a group of support adj matrices - [K,N,N]  k=3
        '''                               
        out = [x]
        
        for i in range(support.shape[0]):
            support_k = support[i,:,:]
            x1 = self.nconv(x,support_k)
            out.append(x1)
            for k in range(2,self.order+1):
                x2 = self.nconv(x1,support_k)
                out.append(x2)
                x1 = x2
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h                        

class nconv_dynamic(nn.Module):
    def __init__(self):
        super(nconv_dynamic,self).__init__()

    def forward(self,x, A):
        '''
        Param x: input feature - torch.Tensor [B,C,N,T]
        Param A: support adj matrices - [B,N,N]
        '''                          
        x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        return x.contiguous()       
class GCN_D(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=2,order=2):
        super(GCN_D,self).__init__()
        self.nconv_dynamic = nconv_dynamic()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        '''
        Param x: input feature - torch.Tensor [B,C,N,T]
        Param support: a group of support adj matrices - [B,K,N,N]  k=3
        '''       
#         assert self.K == support.shape[1]
        out = [x]
#         x1 = self.nconv_dynamic(x,support[:,0,:,:])
        for i in range(support.shape[1]):
            a = support[:,i,:,:]
            x1 = self.nconv_dynamic(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv_dynamic(x1,a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h  
    
'''
spatio_conv_layer

'''
class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c, device, N_adj, gcn_layers, gat_layers, t_channel=10):
        super(spatio_conv_layer, self).__init__()
        self.gcn_layers = gcn_layers
        self.gat_layers = gat_layers

        k = 2 
        self.dropout = 0
        self.N_sta , self.N_dyn = N_adj
        self.gcn_dyn, self.gcn_sta = nn.ModuleList(), nn.ModuleList()
        self.gat_dyn, self.gat_sta = nn.ModuleList(), nn.ModuleList()
        
        for s in range(self.N_sta):
            gat = nn.ModuleList()                      
            gcn = nn.ModuleList()
            for i in range(gat_layers):
                gat.append(GAT(device, in_dim=c, out_dim=t_channel,embed_size=c))
            for i in range(gcn_layers):
                gcn.append(GCN(c_in = c,c_out = c,dropout=0,support_len=k,order=2))       
            self.gcn_sta.append(gcn)
            self.gat_sta.append(gat)

        for d in range(self.N_dyn):
            gat_d = nn.ModuleList()                      
    #         self.gatlayer = GAT(in_dim=c, out_dim=10,embed_size=c)   #demand input [B,C,N,T]
            gcn_d = nn.ModuleList()
            for i in range(gat_layers):
                gat_d.append(GAT_D(device, in_dim=c, out_dim=t_channel,embed_size=c))
            for i in range(gcn_layers):
                gcn_d.append(GCN_D(c_in = c,c_out = c,dropout=0,support_len=k,order=2))       
            self.gcn_dyn.append(gcn_d)
            self.gat_dyn.append(gat_d)            

#         self.gat = nn.ModuleList()                      
#         self.gcn = nn.ModuleList()
#         for i in range(gat_layers):
#             self.gat.append(GAT(in_dim=c, out_dim=10,embed_size=c))   #demand input [B,C,N,T]
#         for i in range(gcn_layers):
#             self.gcn.append(GCN(c_in = c,c_out = c,dropout=0,support_len=k,order=2))        
            
        self.fgat = nn.Linear(c*(self.N_dyn+self.N_sta), c)
        self.fgcn = nn.Linear(c*(self.N_dyn+self.N_sta), c)
        self.mlp = linear(c*2*(self.N_dyn+self.N_sta),c)   # demand [B,C,N,T]
    def forward(self, x, sta_g, sta_support, dyn_g, dyn_support):   # x [B,C,T,N]
        '''
        Param x : input features - torch.Tensor [B,C,T,N]
        Param sta_g: dgl.graph  - torch.Tensor [G,N,N]
        Param sta_support: a group of support adj matrices - [G,K,N,N]  # k=2 or 1, G=4 is the number of statistic graph
        Param dyn_g: dgl.graph  - torch.Tensor [G,B,N,N]
        Param dyn_support: a group of support adj matrices - [G,B,K,N,N]  # k=2 or 1, G=2 is the number of statistic graph  
        '''
#         print('x shape',x.shape)
        
        x_input = x.permute(0,1,3,2)

        h = []
        # statistic gcn and gat
        for s in range(self.N_sta):  
            gcn_sta_out = self.gcn_sta[s][0](x_input,sta_support[s])
            if self.gcn_layers>1:
                for gcn in self.gcn_sta[s][1:]:
                    gcn_sta_out = gcn(gcn_sta_out,sta_support[s])
            h.append(gcn_sta_out)
            gat_sta_out = self.gat_sta[s][0](x_input,sta_g[s])

            if self.gat_layers>1:
                for gat in self.gat_sta[s][1:]:
                    gat_sta_out = gat(gat_sta_out,sta_g[s])
            h.append(gat_sta_out)
        # dynamic gcn and gat
        for d in range(self.N_dyn):
            gcn_dyn_out = self.gcn_dyn[d][0](x_input,dyn_support[d])
            if self.gcn_layers>1:
                for gcn in self.gcn_dyn[d][1:]:
                    gcn_dyn_out = gcn(gcn_dyn_out,dyn_support[d])
            h.append(gcn_dyn_out)
            gat_dyn_out = self.gat_dyn[d][0](x_input,dyn_g[d])
            if self.gat_layers>1:
                for gat in self.gat_dyn[d][1:]:
                    gat_dyn_out = gat(gat_dyn_out,dyn_g[d])
            h.append(gat_dyn_out)             
                
# gat gcn add ：    
#         gat_out = gat_out.permute(0,1,3,2)
#         gcn_out = gcn_out.permute(0,1,3,2)
#         out = gat_out

#  # gat gcn fusion ： 
#         gat_out = gat_out.permute(0,1,3,2)
#         gcn_out = gcn_out.permute(0,1,3,2)
#         gcn , gat = gcn_out.permute(0,3,2,1),gat_out.permute(0,3,2,1)
#         gg = torch.sigmoid(torch.sigmoid(self.fgat(gat) +  self.fgcn(gcn)))
#         out = gg*gat + (1-gg)*gcn
#         out = gcn+gat
#         out = out.permute(0,3,2,1)        
#         out = out + x

# gat gcn mlp:
#         h = torch.cat([gat_out,gcn_out],dim=1)
        h = torch.cat(h,dim=1)  # C*12  
        h = self.mlp(h)   # C*12 --> C
        h = F.dropout(h, self.dropout, training=self.training)
        out = h.permute(0,1,3,2)

        return out

'''
st_conv_block
'''
class st_conv_block(nn.Module):
    def __init__(self, ks, kt, n, c, p, device, N_adj, gcn_layers, gat_layers, t_channel=10):
        super(st_conv_block, self).__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
        self.sconv = spatio_conv_layer(ks, c[1], device, N_adj, gcn_layers, gat_layers,t_channel)
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])

        self.dropout = nn.Dropout(p)

    def forward(self, x, sta_g, sta_support, dya_g, dya_support):
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1, sta_g, sta_support, dya_g, dya_support)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)        
        return self.dropout(x_ln)
      
'''
output_layer
'''
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

'''
TCN_DGLGCN
'''                          
class TCN_DGLGCN(nn.Module):
    def __init__(self, ks, kt, bs_hour, bs_day, T, n, p, device, N_adj:tuple, gcn_layers, gat_layers):
        super(TCN_DGLGCN, self).__init__()
        self.st_conv_h1 = st_conv_block(ks, kt, n, bs_hour[0], p, device, N_adj, gcn_layers, gat_layers, t_channel=10)
        self.st_conv_h2 = st_conv_block(ks, kt, n, bs_hour[1], p, device, N_adj, gcn_layers, gat_layers, t_channel=6)
        self.st_conv_d1 = st_conv_block(ks, kt, n, bs_day[0], p, device, N_adj, gcn_layers, gat_layers, t_channel=10)
        self.st_conv_d2 = st_conv_block(ks, kt, n, bs_hour[1], p, device, N_adj, gcn_layers, gat_layers, t_channel=6)
        
#         self.output = output_layer(bs[1][2], T - 4 * (kt - 1), n)
        self.output = output_layer(bs_day[1][2], 8, n)

    def forward(self, x, sta_g, sta_support, dyn_g, dyn_support):
#         print('x shape', x.shape)
        x_h = x[:,0:3,:,:]   # IN+OUT+TIME
        x_d = x[:,3:,:,:]    # HA FIVE DAYS
        x_st_h1 = self.st_conv_h1(x_h, sta_g, sta_support, dyn_g, dyn_support)
        x_st_h2 = self.st_conv_h2(x_st_h1, sta_g, sta_support, dyn_g, dyn_support)
        x_st_d1 = self.st_conv_d1(x_d, sta_g, sta_support, dyn_g, dyn_support)
        x_st_d2 = self.st_conv_d2(x_st_d1, sta_g, sta_support, dyn_g, dyn_support)

        
        x_d_h = torch.cat((x_st_h2 ,x_st_d2), dim=2)
        
#         return torch.relu(self.output(x_st2))
        return self.output(x_d_h)

'''
adj preprocess
'''                          
def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()
def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian
def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()                                               
def load_adj(pkl_filename, adjtype):
    adj_mx = np.array(pd.read_csv(pkl_filename)).astype(np.float32)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
#         adj = np.array(adj)
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
#     print('adj shape : ',adj.shape)
    return adj
def single_adj_process(adj, adjtype):
    filename = './data/adj/W_'+str(adj)+'.csv'
    adj_mx = np.array(pd.read_csv(filename)).astype(np.float32)
    distances = adj_mx[~np.isinf(adj_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(adj_mx / std))    
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    adj = np.array(adj)
#     print('adj shape : ',adj.shape)
    return adj
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
        adj_max = np.max(W)*1.5
        
        W = W/10000
        W2 = W * W
        W_mask = (np.ones([n, n]) - np.identity(n))
        W_mask2 = W_mask*(np.exp(-W2) < 1)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W
                          
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
    CHANNEL = 8
    # ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 32, 64], [64, 32, 128]], TIMESTEP_IN, N_NODE, 0
    ks, kt, bs_hour, bs_day, T, n, p = 3, 3, [[3, 16, 64], [64, 16, 64]], [[5, 16, 64], [64, 16, 64]], INPUT_STEP, N_NODE, 0
    # dgl graph                       
    A = pd.read_csv(ADJPATH).values
    G = get_normalized_adj(A)
#     g = sp.csr_matrix(G)
#     ## 方式2: 使用稀疏矩阵进行构造
#     g2 = dgl.from_scipy(g)
        
    sta_g = np.array(4*[G])
    dyn_g = np.array(2*[np.array(8*[G])])                                    
    sta_g = torch.Tensor(sta_g)
    dyn_g = torch.Tensor(dyn_g)                                        
    #gcn graph                      
    adj_mx =load_adj(ADJPATH,ADJTYPE)  
    supports = [i for i in adj_mx]
    supports = np.array(supports)  #[2,82,81]
#     supports = np.array(2*[supports])
    sta_supports = np.array(4*[supports])   #[4,2,82,81]
    dyn_supports = np.array(8*[supports])
    dyn_supports = np.array(2*[dyn_supports])  #[2,8,2,82,81]
    sta_supports = torch.Tensor(sta_supports).to(device)
    dyn_supports = torch.Tensor(dyn_supports).to(device)
    
    X = torch.Tensor(torch.randn(8,8,12,81)).to(device)
    model = TCN_DGLGCN(ks, kt, bs_hour, bs_day, T, n, p, device, N_adj=[4,2], gcn_layers=2, gat_layers=2).to(device)
    model(X, sta_g, sta_supports, dyn_g, dyn_supports)
    print_params('TCN_DGLGCN',model)
#     summary(model, (CHANNEL, INPUT_STEP, N_NODE), device=device)
#     writer = SummaryWriter(log_dir='logs')
#     # 将网络net的结构写到logs里：
#     data = torch.rand(8,2,12,81).to(device)
#     writer.add_graph(model,input_to_model=(data,))
    
if __name__ == '__main__':
    main()  
