import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
import scipy.sparse as sp

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class PatternFlowBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_attention_layers):
        super(PatternFlowBranch, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attentions = nn.ModuleList([nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads) for _ in range(num_attention_layers)])
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_attention_layers)])
        self.norm_out = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        for attn, norm in zip(self.attentions, self.norms):
            x1 = x.unsqueeze(1)  
            attn_output, _ = attn(x1, x1, x1)
            attn_output = attn_output.squeeze(1)  
            x = norm(attn_output + x)  
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.norm_out(x)
        return x

class PatternFlowNet(nn.Module):
    def __init__(self, num_branches, input_dim, hidden_dim, branch_output_dim, final_output_dim, num_heads):
        super(PatternFlowNet, self).__init__()
        self.branches = nn.ModuleList([PatternFlowBranch(input_dim, hidden_dim, branch_output_dim, num_heads, 4) for _ in range(num_branches)])
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=branch_output_dim, nhead=num_heads),
            num_layers=4
        )
        self.final_norm = nn.LayerNorm(branch_output_dim * num_branches)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = DeepFeedForward(256, 128)
        self.decoder_s = nn.Linear(128, 128)
        self.decoder_t = nn.Linear(128, 128)
        self.feature = None
        self.sageconv = GraphSAGEModel(branch_output_dim * num_branches, 512, 256)
        adj_matrix = np.load('Data/adj_matrix.npy')
        adj_matrix_sparse = sp.coo_matrix(adj_matrix)
        edge_index, _ = from_scipy_sparse_matrix(adj_matrix_sparse)
        self.edge_index = edge_index

    def forward(self, graphs):
        branch_outputs = [branch(graph) for branch, graph in zip(self.branches, graphs)]  # (batch_size, branch_output_dim)
        branch_outputs = torch.stack(branch_outputs, dim=1)  # (batch_size, num_branches, branch_output_dim)
        
        branch_outputs = branch_outputs.permute(1, 0, 2)  # (num_branches, batch_size, branch_output_dim)
        transformer_out = self.transformer_encoder(branch_outputs)
        transformer_out = transformer_out.permute(1, 0, 2)  # (batch_size, num_branches, branch_output_dim)
        concatenated = transformer_out.contiguous().view(transformer_out.size(0), -1)  # (batch_size, branch_output_dim * num_branches)
        
        out = self.final_norm(concatenated)
        out = self.sageconv(out, self.edge_index)

        out = self.fc(out)
        self.feature = out
        
        out_s = self.decoder_s(out)
        out_t = self.decoder_t(out)
        
        return out_s, out_t

    def get_features(self):
        return self.feature

class DeepFeedForward(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepFeedForward, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
        )
        self.output = None

    def forward(self, x):
        output = self.model(x)
        self.output = output
        return output

    def get_output(self):
        return self.output
