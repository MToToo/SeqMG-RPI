import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_max_pool
from torch_geometric.utils import add_self_loops
from src.kan_layer import NaiveFourierKANLayer as KANLayer

class ProteinFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, grid_feat=200, heads=8, use_bias=False):  # grid_feat:Grid size for Fourier KAN
        super(ProteinFeatureExtractor, self).__init__()

        self.lin_in = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.kan1 = KANLayer(hidden_dim, hidden_dim, grid_feat, addbias=use_bias)
        self.kan2 = KANLayer(hidden_dim, hidden_dim, grid_feat, addbias=use_bias)
        self.GAT = GATConv(hidden_dim, hidden_dim // heads, heads=heads)
        self.lin_out = nn.Linear(hidden_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_weight, batch):
        # If there is no edge, add a self-loop
        if edge_index.size(1) == 0:
            print("Warning: No edges found in the graph, adding self-loops.")
            edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, num_nodes=x.size(0))
        x = self.lin_in(x)
        x = self.GAT(x, edge_index, edge_attr=edge_weight)
        x = self.kan1(x)
        x = self.GAT(x, edge_index, edge_attr=edge_weight)
        x = self.kan2(x)
        x = global_max_pool(x, batch)
        x = self.lin_out(x)
        return x
