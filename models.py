import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from data_utils import random_neighbor_sampling,load_graph_data,load_path_data
import numpy as np
import scipy.sparse as sp

class CustomMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, **kwargs):
        # 调用父类的构造函数
        super(CustomMultiheadAttention, self).__init__(embed_dim, num_heads, **kwargs)
        self.gate_dim = embed_dim  # 门机制的维度，如果为 None 则不启用门
        self.gate_fc = nn.Linear(embed_dim, embed_dim)
        self.two_dim_bias = torch.tensor(load_path_data(), dtype=torch.float32)
        self.a = nn.Parameter(torch.tensor(0.5))  # 默认值为 0.5
        self.b = 1.0 - self.a  # b 是 1 - a，保证 a + b = 1


    def forward(self,query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):

        # 如果启用了门机制
        gate = torch.sigmoid(self.gate_fc(query))  # (seq_len, batch_size, gate_dim)
        # 将门控制系数和原始输入相乘（按元素相乘）
        # query = query * gate  # query 的每个元素会按门系数缩放

        # 调用父类的 forward 方法计算标准的多头注意力
        attn_output, attn_output_weights = super().forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False)

        # 如果使用了门机制，输出时乘回门控制系数
        attn_output = attn_output * gate  # 将门控制系数乘回到输出上
        attn_output_weights = attn_output_weights*self.a + self.two_dim_bias*self.b

        return attn_output, attn_output_weights




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

class IntraGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_attention_layers):
        super(IntraGraph, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attentions = nn.ModuleList([CustomMultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads) for _ in range(num_attention_layers)])
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

class InterGraph(nn.Module):
    def __init__(self, branch_output_dim, num_heads, num_layers):
        super(InterGraph, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=branch_output_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.final_norm = nn.LayerNorm(branch_output_dim * num_heads)
        
    def forward(self, branch_outputs):
        # branch_outputs: (batch_size, num_branches, branch_output_dim)
        branch_outputs = branch_outputs.permute(1, 0, 2)  # (num_branches, batch_size, branch_output_dim)
        transformer_out = self.transformer_encoder(branch_outputs)
        transformer_out = transformer_out.permute(1, 0, 2)  # (batch_size, num_branches, branch_output_dim)
        concatenated = transformer_out.contiguous().view(transformer_out.size(0), -1)  # (batch_size, branch_output_dim * num_branches)
        out = self.final_norm(concatenated)
        return out

class MSIN(nn.Module):
    def __init__(self, num_branches, input_dim, hidden_dim, branch_output_dim, final_output_dim, num_heads):
        super(MSIN, self).__init__()
        self.branches = nn.ModuleList([IntraGraph(input_dim, hidden_dim, branch_output_dim, num_heads, 4) for _ in range(num_branches)])
        self.inter_graph = InterGraph(branch_output_dim, num_heads, num_layers=4)
        self.dropout = nn.Dropout(p=0.5)
        self.sageconv = GraphSAGEModel(branch_output_dim * num_branches, final_output_dim*2, final_output_dim)

        self.fc = DeepFeedForward(final_output_dim, final_output_dim)
        self.decoder_s = nn.Linear(final_output_dim, final_output_dim)
        self.decoder_t = nn.Linear(final_output_dim, final_output_dim)
        self.feature = None
        
        edge_index, edge_attr = load_graph_data()
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def forward(self, graphs):
        branch_outputs = [branch(graph) for branch, graph in zip(self.branches, graphs)]  # (batch_size, branch_output_dim)
        branch_outputs = torch.stack(branch_outputs, dim=1)  # (batch_size, num_branches, branch_output_dim)
        
        out = self.inter_graph(branch_outputs)
        out = self.sageconv(out, random_neighbor_sampling(self.edge_index,self.edge_attr))

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


