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
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.W = nn.Parameter(torch.empty(size=(2*in_channels, embed_dim)))
#         self.W = nn.Parameter(torch.empty([embed_dim, in_channels]), requires_grad=True)
        self.a = nn.Parameter(torch.empty(embed_dim,1), requires_grad=True)
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
        print('0 input x shape :',x.shape)  # [B, N, in_channels]
        hidden = torch.matmul(x,self.W)
        print('1 hidden shape :',hidden.shape)  # [B, N, embed_dim]
        B = x.shape[0]
        N = x.shape[1]
        x_repeated_in_chunks = x.repeat_interleave(N, dim=1)
        x_repeated_alternating = x.repeat(1, N, 1)
        print(' x_repeated_alternating shape :',x_repeated_alternating.shape)   # [B, N*N, channels]  
        all_combinations_matrix = torch.cat([x_repeated_in_chunks, x_repeated_alternating], dim=1).view(B, N, N, 2 * self.in_channels)
        print(' all_combinations_matrix shape :',all_combinations_matrix.shape)     # [B, N, N, 2 * channels]
        wx = torch.matmul(x,self.W)
        # leaky ReLU
        attn = self.leaky_relu(wx)   # [B, N, N, embed_dim]  
        print('3 leaky ReLU attn shape :',attn.shape)
        attn= torch.matmul(attn,self.a.to(self.device)).squeeze(-1) # [B, N, N]  
        print('2 attn shape:',attn.shape)      
        # adj
        if adj is not None:
            attn += torch.where(adj > 0.0, 0.0, -1e12)
        # softmax
        attention = F.softmax(attn, dim=-1)  # [B, N, N]
        print('4 attention softmax: ',attention.shape)
        # dropout
        attention = F.dropout(attention, self.dropout, training=self.training)
        output = torch.matmul(attention,wx)  # [B, N, embed_dim]
        print('4 output: ',output.shape)
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
                 n_heads=8, dropout=0.1, alpha=0.2, bias=True, aggregate='concat'):
        """
        Args:
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
            output = sum([attn(x, adj) for attn in self.attns]) / len(self.attns)
        output = F.dropout(output, self.dropout, training=self.training)
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
    model = GAT(in_channels=C,embed_dim=32,device= device).to(device)
#     model(X,adj)
    print_params('GAT1',model)
    summary(model, [(81,C),(81,81)], device=device)
    
if __name__ == '__main__':
    main()   