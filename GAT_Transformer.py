# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:28:06 2020
@author: wb
"""
import sys
import torch
import torch.nn as nn
# from GCN_models import GCN
# from One_hot_encoder import One_hot_encoder
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from Param import *

class GraphAttention(nn.Module):
    def __init__(self, in_channels, out_channels, device, dropout=0.1, alpha=0.2, bias=True):
        super(GraphAttention, self).__init__()
        self.W = nn.Parameter(torch.empty(size=(in_channels, out_channels)))
#         self.W = nn.Parameter(torch.empty([out_channels, in_channels]), requires_grad=True)
        self.a = nn.Parameter(torch.empty(out_channels,2 ), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True) if bias else None
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.device  = device
        
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): shape in [batch, nodes, in_channels]
            mask (torch.Tensor, optional): shape is [batch, nodes, nodes]. Defaults to None.
        Returns:
            [torch.Tensor]: shape is [batch, nodes, out_channels]
        """
        print('cuda device is :',self.device)
        print('0 input x shape :',x.shape)
        hidden = torch.matmul(x,self.W)
        print('1 hidden shape :',hidden.shape)  # [B, N, out_channels]
        
        print(' hidden * a shape: ',torch.matmul(hidden,self.a.to(self.device)).shape)   # [B,N,2]
        attn1, attn2 = torch.unbind(torch.matmul(hidden,self.a.to(self.device)), -1)  # [B, N]   unbind 把最后一维度2分开，返回给attn1和attn2
        print('2 attn1 shape:',attn1.shape)      
        # [batch, nodes, 1] + [batch, 1, nodes] => [B, N, N]
        attn = attn1.unsqueeze(-1) + attn2.unsqueeze(1)
        # leaky ReLU
        attn = self.leaky_relu(attn)
        print('3 leaky ReLU attn shape :',attn.shape)
        # adj
        if adj is not None:
            attn += torch.where(adj > 0.0, 0.0, -1e12)
        # softmax
        attn = torch.softmax(attn, dim=-1)  # [B, N, N]
        # dropout
        attn, hidden = self.dropout(attn), self.dropout(hidden)
        output = torch.matmul(attn,hidden)  # [B, N, out_channels]
        # add bias
        if self.bias is not None:
            output += self.bias.to(self.device)
        return output  # [B, N, out_channels]


class GAT(nn.Module):
    """
    Graph Attention Network
    """

    def __init__(self,  in_channels, out_channels, device, n_heads=8, dropout=0.1, alpha=0.2, bias=True, aggregate='concat',):
        """
        Args:
            in_channels ([type]): input channels
            out_channels ([type]): output channels
            n_heads (int, optional): number of heads. Defaults to 64.
            dropout (float, optional): dropout rate. Defaults to .1.
            alpha (float, optional): leaky ReLU negative_slope. Defaults to .2.
            bias (bool, optional): use bias. Defaults to True.
            aggregate (str, optional): aggregation method. Defaults to 'concat'.
        """
        super(GAT, self).__init__()
        assert aggregate in ['concat', 'average']
        self.attns = nn.ModuleList([
            GraphAttention( in_channels, out_channels, device, dropout=dropout, alpha=alpha, bias=bias)
            for _ in range(n_heads)
        ])
        self.aggregate = aggregate

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): shape is [batch, nodes, in_channels]
            adj (torch.Tensor, optional): shape is [batch, nodes, nodes]. Defaults to None.
        """
        adj = adj.long()
        if self.aggregate == 'concat':
            output = torch.cat([attn(x, adj) for attn in self.attns], dim=-1)
            print('output shape:',output.shape)
        else:
            output = sum([attn(x, adj) for attn in self.attns]) / len(self.attns)
        return torch.relu(output)



class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 可能没有
        '''
        B, n_heads, len1, len2, d_k = Q.shape 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        # scores : [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context

class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
            
        # 用Linear来做投影矩阵    
        # 但这里如果是多头的话，是不是需要声明多个矩阵？？？

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4) # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V) #[B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4) #[B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim) # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context) # [batch_size, len_q, d_model]
        return output 

class Transformer_EncoderBlock(nn.Module):
    def __init__(self, embed_size, time_num, heads ,forward_expansion, gpu, dropout ):
        super(Transformer_EncoderBlock, self).__init__()
        
        # Temporal embedding One hot
        self.time_num = time_num
#         self.one_hot = One_hot_encoder(embed_size, time_num)          # temporal embedding选用one-hot方式 或者
        self.temporal_embedding = nn.Embedding(time_num, embed_size)  # temporal embedding选用nn.Embedding
        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.gpu = gpu
    def forward(self, value, key, query):
        B, N, T, C = query.shape
        
#         D_T = self.one_hot(t, N, T)                          # temporal embedding选用one-hot方式 或者
        D_T = self.temporal_embedding(torch.arange(0, T).to(self.gpu))    # temporal embedding选用nn.Embedding
        D_T = D_T.expand(B, N, T, C)
        print('query shape:',query.shape)
        print('D_T shape:',D_T.shape)
        # temporal embedding加到query。 原论文采用concatenated
        query = query + D_T  
        print('query + D_T shape:',query.shape)
        attention = self.attention(query, query, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

### TBlock

class TTransformer_EncoderLayer(nn.Module):
    def __init__(self, embed_size, time_num, heads ,forward_expansion, gpu, dropout):
        super(TTransformer_EncoderLayer, self).__init__()
#         self.STransformer = STransformer(embed_size, heads, adj, cheb_K, dropout, forward_expansion)
        self.Transformer_EncoderBlock = Transformer_EncoderBlock(embed_size, time_num, heads ,forward_expansion, gpu, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query):
    # value,  key, query: [N, T, C] [B, N, T, C]
        # Add skip connection,run through normalization and finally dropout
        x1 = self.norm1(self.Transformer_EncoderBlock(value, key, query) + query) #(B, N, T, C)
        x2 = self.dropout(x1) 
#         x1 = self.norm1(self.STransformer(value, key, query) + query) #(B, N, T, C)
#         x2 = self.dropout( self.norm2(self.TTransformer(x1, x1, x1, t) + x1) ) 
        return x2

### Encoder
class Encoder(nn.Module):
    # 堆叠多层 ST-Transformer Block
    def __init__(
        self,embed_size,num_layers,time_num,heads,forward_expansion,gpu,dropout):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.gpu = gpu
        self.layers = nn.ModuleList([ TTransformer_EncoderLayer(embed_size, time_num, heads ,forward_expansion, gpu, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
    # x: [N, T, C]  [B, N, T, C]
        out = self.dropout(x)        
        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            out = layer(out, out, out)
        return out     
    
### Transformer   
class T_Transformer_block(nn.Module):
    def __init__(self,embed_size,num_layers,time_num,heads,forward_expansion, gpu,dropout):
        super(T_Transformer_block, self).__init__()
        self.encoder = Encoder(embed_size,num_layers,time_num,heads,forward_expansion,gpu,dropout)
        self.gpu = gpu

    def forward(self, src): 
        ## scr: [N, T, C]   [B, N, T, C]
        enc_src = self.encoder(src) 
        return enc_src # [B, N, T, C]

### ST Transformer: Total Model

class T_Transformer(nn.Module):
    def __init__(
        self, in_channel, embed_size, time_num, num_layers, T_dim, output_T_dim, heads, forward_expansion, gpu, dropout):        
        super(T_Transformer, self).__init__()
        self.T_dim=T_dim
        self.forward_expansion = forward_expansion
        # 第一次卷积扩充通道数
#         self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.gatlayer = GAT(in_channels = in_channel, out_channels = 8, device = gpu)
        
        self.T_Transformer_block = T_Transformer_block(embed_size, num_layers, time_num,heads, forward_expansion, gpu, dropout)

        # 缩小时间维度。  例：T_dim=12到output_T_dim=3，输入12维降到输出3维
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)  
        # 缩小通道数，降到1维。
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x, adj):
        # input x shape[B, Ci, N, T]  Ci  = 2
        # C:通道数量。  N:传感器数量。  T:时间数量  
        GAT_lst = list()
        for i in range(self.T_dim):
            in_GAT = x[:,:,:,i].permute(0,2,1)   # [B,Ci,N]   ---[B,N,Ci] 
            out_GAT = self.gatlayer(in_GAT,adj)
            GAT_lst.append(out_GAT)
        GAT_feature = torch.stack(GAT_lst, dim=2) #[B,N,T,Co] Co=embed_size=64
        print('GAT_feature: shape',GAT_feature.shape)    #[B,N,T,Co]  Co=embed_size=64
#         x = x.unsqueeze(0)   
#         input_Transformer = self.conv1(x)     #    x shape[B, 2, N, T]   --->    input_Transformer shape： [B, C = embed_size = 64, N, T] 
#         input_Transformer = GAT_feature.permute(0, 2, 3, 1)    # 等号左边 input_Transformer shape: [B, N, T, C]
        output_Transformer = self.T_Transformer_block(GAT_feature)  # 等号左边 output_Transformer shape: # [B, N, T, C]
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)   # 等号左边 output_Transformer shape: [B, T, N, C]
        
#         output_Transformer = output_Transformer.unsqueeze(0)     
        out = self.relu(self.conv2(output_Transformer))    # 等号左边 out shape: [B, output_T_dim = PRED_STEP, N, C]        
        out = out.permute(0, 3, 2, 1)           # 等号左边 out shape: [B, C, N, output_T_dim = PRED_STEP]
        out = self.conv3(out)                   # 等号左边 out shape: [B, 1, N, output_T_dim = PRED_STEP]   
        out = out.squeeze(1)
           
        return out      #[B, N, output_dim]
        # return out shape: [B, N, output_dim]
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
    model = T_Transformer(in_channel = 2, embed_size = 64, time_num = 82 , num_layers = 3, T_dim = INPUT_STEP, output_T_dim = PRED_STEP, heads = 8, forward_expansion = 4, gpu = device, dropout = 0).to(device)
    print_params('GAT_Transformer',model)
    summary(model, [(2, N_NODE, INPUT_STEP),(N_NODE,N_NODE)], device=device)
    print_params('GAT_Transformer',model)
    
if __name__ == '__main__':
    main()
