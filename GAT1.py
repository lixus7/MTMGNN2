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
        self.bias = nn.Parameter(torch.empty(embed_dim), requires_grad=True) if bias else None
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.device  = device
        
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
        print('cuda device is :',self.device)
        print('0 input x shape :',x.shape) # [B, N, C]
        hidden = torch.matmul(x,self.W)
        print('1 hidden shape :',hidden.shape)  # [B, N, embed_dim]
        B = hidden.shape[0]
        N = hidden.shape[1]
        hidden_repeated_in_chunks = hidden.repeat_interleave(N, dim=1)
        hidden_repeated_alternating = hidden.repeat(1, N, 1)
        print(' hidden_repeated_alternating shape :',hidden_repeated_alternating.shape) 
        all_combinations_matrix = torch.cat([hidden_repeated_in_chunks, hidden_repeated_alternating], dim=1).view(B, N, N, 2 * self.embed_dim)
        print(' all_combinations_matrix shape :',all_combinations_matrix.shape) 
        attn= torch.matmul(all_combinations_matrix,self.a.to(self.device)).squeeze(-1) # [B, N, N]  
        print('2 attn shape:',attn.shape)      
        # leaky ReLU
        attn = self.leaky_relu(attn)   # [B, N, N]  
        print('3 leaky ReLU attn shape :',attn.shape)
        # adj
        zero_vec = -9e15*torch.ones_like(attn)
        attn = torch.where(adj > 0, attn, zero_vec)
#         if adj is not None:
#             attn += torch.where(adj > 0.0, 0.0, -1e12)
        # softmax
        attention = F.softmax(attn, dim=-1)  # [B, N, N]
        print('4 attention softmax: ',attention.shape)
        # dropout
        attention = F.dropout(attention, self.dropout, training=self.training)
        output = torch.matmul(attention,hidden)  #   atten [B,N,N] *  hidden [B,N,embed_dim = 64]     ---->  output [B, N, embed_dim]     feature * att_score
        print('5 output: ',output.shape)
        # add bias
        if self.bias is not None:
            output += self.bias.to(self.device)
            print('return output: ',output.shape)
        return output  # [B, N, embed_dim]


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
        adj = adj.long()
        x = F.dropout(x, self.dropout, training=self.training)
        if self.aggregate == 'concat':
            output = torch.cat([attn(x, adj) for attn in self.attns], dim=-1)
            print('output shape:',output.shape)    #  [2, 81, 512]
        else:
            output = torch.mean(torch.stack([attn(x, adj) for attn in self.attns]),dim=0)  #  [2, 81, 64]
            print('output shape:',output.shape)
#             output = sum([attn(x, adj) for attn in self.attns]) / len(self.attns)
        output = F.dropout(output, self.dropout, training=self.training)   #  [2, 81, 64]
        print('return out :',F.elu(output).shape)
        return F.elu(output)
    
    
def print_params(model_name, model):
    param_count=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += param.numel()
    print(f'{model_name}, {param_count} trainable parameters in total.')
    return        
def main():
    C = 2
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '7'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    adj = torch.rand(81, 81)
    adj = torch.Tensor(adj)
    adj = adj.to(device=device, dtype=torch.float64)
    X = torch.Tensor(torch.randn(32,81,64)).to(device)
    model = GAT(device= device,in_channels=C,embed_dim=32,aggregate='average').to(device)
#     model(X,adj)
    print_params('GAT1',model)
    summary(model, [(81,C),(81,81)], device=device)
    
if __name__ == '__main__':
    main()   