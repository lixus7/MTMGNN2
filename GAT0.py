import torch
from torch import nn
import sys
import scipy.sparse as sp
import numpy as np
from torchsummary import summary

class GraphAttention(nn.Module):
    def __init__(self, device,in_channels, out_channels, dropout=0.1, alpha=0.2, bias=True):
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
        print('0 input W shape :',self.W.shape)
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

    def __init__(self, device ,in_channels, out_channels,
                 n_heads=8, dropout=0.1, alpha=0.2, bias=True, aggregate='concat'):
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
            GraphAttention(device , in_channels, out_channels, dropout=dropout, alpha=alpha, bias=bias)
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
        return torch.relu(output)   # [2, 81, 64]
    
    
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
    model = GAT(in_channels=C,out_channels=8,device= device).to(device)
#     model(X,adj)
    print_params('GAT0',model)
    summary(model, [(81,C),(81,81)], device=device)
    
if __name__ == '__main__':
    main()   