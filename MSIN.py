import torch
import os
import csv
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tasks import do_tasks
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class PatternGraphBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_attention_layers):
        super(PatternGraphBranch, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attentions = nn.ModuleList([nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads) for _ in range(num_attention_layers)])
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_attention_layers)])
        self.norm_out = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 第一层全连接和ReLU激活
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 多层自注意力机制
        for attn, norm in zip(self.attentions, self.norms):
            x1 = x.unsqueeze(1)  # (batch_size, seq_len, embed_dim)
            attn_output, _ = attn(x1, x1, x1)
            attn_output = attn_output.squeeze(1)  # (batch_size, embed_dim)
            x = norm(attn_output + x)  # 残差连接和LayerNorm

        # 第二层全连接和ReLU激活
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.norm_out(x)
        return x


class PatternGraphNet(nn.Module):
    def __init__(self, num_graphs, input_dim, hidden_dim, branch_output_dim, final_output_dim, num_heads):
        super(PatternGraphNet, self).__init__()
        self.branches = nn.ModuleList([PatternGraphBranch(input_dim, hidden_dim, branch_output_dim, num_heads, 4) for _ in range(num_graphs)])
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=branch_output_dim, nhead=8),
            num_layers=4
        )
        self.final_norm = nn.LayerNorm(branch_output_dim * num_graphs)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = DeepFc(256, 128)
        self.decoder_s = nn.Linear(128, 128)
        self.decoder_t = nn.Linear(128, 128)
        self.feature = None
        self.sageconv = GraphSAGE(branch_output_dim * num_graphs, 512, 256)
        adj_matrix = np.load('adj_matrix.npy')
        adj_matrix_sparse = sp.coo_matrix(adj_matrix)
        edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix_sparse)
        self.edge_index = edge_index

    def forward(self, graphs):
        branch_outputs = [branch(graph) for branch, graph in zip(self.branches, graphs)]  # (batch_size, branch_output_dim)
        branch_outputs = torch.stack(branch_outputs, dim=1)  # (batch_size, num_graphs, branch_output_dim)
        
        # Transformer处理时序数据
        branch_outputs = branch_outputs.permute(1, 0, 2)  # (num_graphs, batch_size, branch_output_dim)
        transformer_out = self.transformer_encoder(branch_outputs)
        transformer_out = transformer_out.permute(1, 0, 2)  # (batch_size, num_graphs, branch_output_dim)
        concatenated = transformer_out.contiguous().view(transformer_out.size(0), -1)  # (batch_size, branch_output_dim * num_graphs)
        
        # 残差连接和LayerNorm
        out = self.final_norm(concatenated)
        
        # GraphSAGE处理
        out = self.sageconv(out, self.edge_index)
        # 全连接层处理
        out = self.fc(out)
        self.feature = out
        
        # 解码器
        out_s = self.decoder_s(out)
        out_t = self.decoder_t(out)
        
        return out_s, out_t

    def out_feature(self):
        return self.feature

class DeepFc(nn.Module):
    def __init__(self, input_dim, output_dim):
        # 输入层，隐藏层*2,输出层.隐藏层节点数目为输入层两倍
        super(DeepFc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.3, inplace=True), )
        self.output = None

    def forward(self, x):
        output = self.model(x)
        self.output = output
        return output

    def out_feature(self):
        return self.output



def inner_product(mat_1, mat_2):
    n, m = mat_1.shape 
    mat_expand = torch.unsqueeze(mat_2, 0)  
    mat_expand = mat_expand.expand(n, n, m)  
    mat_expand = mat_expand.permute(1, 0, 2)  
    inner_prod = torch.mul(mat_expand, mat_1)  
    inner_prod = torch.sum(inner_prod, axis=-1)  
    return inner_prod


def _mob_loss(s_embeddings, t_embeddings, mob):
    inner_prod = inner_product(s_embeddings, t_embeddings)
    softmax1 = nn.Softmax(dim=-1)
    phat = softmax1(inner_prod)
    loss = torch.sum(-torch.mul(mob, torch.log(phat+0.0001)))
    inner_prod = pairwise_inner_product(t_embeddings, s_embeddings)
    softmax2 = nn.Softmax(dim=-1)
    phat = softmax2(inner_prod)
    loss += torch.sum(-torch.mul(torch.transpose(mob, 0, 1), torch.log(phat+0.0001)))
    return loss


class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()

    def forward(self, out1, out2, label):
        mob_loss = _mob_loss(out1, out2, label)
        loss = mob_loss
        return loss


def train_model(input_tensor, label, criterion=None, model=None):
    road = np.load('NewData/road_matrix.npy')
    input_tensor.append(torch.tensor(road, dtype=torch.float))
    epochs = 1800
    learning_rate = 0.0005
    weight_decay = 5e-4

    num_graphs = 8
    input_dim = 69  # 输入特征维度
    hidden_dim = 128  # 隐藏层特征维度
    branch_output_dim = 120  # 每个分支的输出特征维度
    final_output_dim = 128  # 最终输出特征维度
    num_heads = 8  # 多头注意力机制中的头数
    if criterion is None:
        criterion = SimLoss()
    if model is None:
        # model = MGFN(graph_num=7, node_num=180, output_dim=emb_dim)
        model = PatternGraphNet(num_graphs, input_dim, hidden_dim, branch_output_dim, final_output_dim, num_heads)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        # out_s, out_t = model(input_tensor)
        s_out, t_out = model(input_tensor)
        loss = criterion(s_out, t_out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch > 1750:
            print("Epoch {}, Loss {}".format(epoch, loss.item()))
            embs = model.out_feature()
            embs = embs.detach().numpy()
            pop_mae, pop_rmse, pop_r2, check_mae, check_rmse, check_r2, nmi, ars = do_tasks(embs)

            # file_exists = os.path.isfile('results.csv')
            
            # 保存结果到 CSV 文件
            # with open('results.csv', 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     if not file_exists:
            #         writer.writerow(columns)
            #     writer.writerow(results)



if __name__ == '__main__':
    mob_pattern = np.load("./Data/mob_pattern.npy")
    pattern_list = [torch.tensor(mob_pattern[i], dtype=torch.float) for i in range(mob_pattern.shape[0])]
    mob_adj = np.load("./Data/mobility.npy")
    mob_pattern = torch.Tensor(mob_pattern)
    mob_adj = torch.Tensor(mob_adj)
    train_model(pattern_list, mob_adj)