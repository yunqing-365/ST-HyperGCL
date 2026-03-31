# 文件路径: models/st_hypergcl.py

import torch
import torch.nn as nn
import torch.nn.functional as F

#残差图神经网络 (空间聚合)

class ResidualGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualGNN, self).__init__()
        self.fc_self = nn.Linear(input_dim, hidden_dim)
        self.fc_neigh = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x, adj):
        x_neigh = torch.matmul(adj, x)
        h = self.fc_self(x) + self.fc_neigh(x_neigh)
        return F.relu(h)

# 无监督图对比学习 (GCL - 去噪)

def supervised_contrastive_loss(z1, z2, labels, temperature=0.5):
    """
    有监督对比学习 (SupCon):
    拉近标签相同的企业表征（违规找违规，健康找健康），推开标签不同的企业。
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # 将两个视图拼接 [2N, D]
    z = torch.cat([z1, z2], dim=0)
    y = torch.cat([labels, labels], dim=0).view(-1, 1)
    
    # 计算相似度矩阵
    sim_matrix = torch.matmul(z, z.t()) / temperature
    
    # 构造掩码 (Mask)：相同标签的位置为 1，不同标签为 0
    mask = torch.eq(y, y.t()).float()
    
    # 消除对角线（不与自己对比）
    mask.fill_diagonal_(0)
    
    # 计算 InfoNCE (仅针对正样本)
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    # 掩码求和取平均
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss

# 架构 (GNN + GRU + GCL)

class UltimateRiskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(UltimateRiskModel, self).__init__()
        self.gnn = ResidualGNN(input_dim, hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj_list):
        T = len(adj_list)
        num_nodes = x.shape[0]
        hidden_dim = self.gnn.fc_self.out_features
        
        temporal_features = torch.zeros(num_nodes, T, hidden_dim)
        
        # 1. 空间切片：每年单独跑一次 GNN (结合静态特征 x)
        for t in range(T):
            node_emb = self.gnn(x, adj_list[t])
            temporal_features[:, t, :] = node_emb
            
        # 2. 时序演化：送入 GRU (此处已被消融，改为时间维度平均池化)
        # gru_out, h_n = self.gru(temporal_features)
        # final_state = gru_out[:, -1, :] # 取最后一年状态
        
        # 在时间维度(dim=1)上求平均，跨越 9 年的静态池化
        final_state = torch.mean(temporal_features, dim=1)
        final_state = self.dropout(final_state)
        
        # 3. 生成两个对比视图 (用于 GCL)
        view1 = F.dropout(final_state, p=0.1, training=self.training)
        view2 = F.dropout(final_state, p=0.1, training=self.training)
        
        # 4. 输出分类概率
        logits = self.classifier(final_state)
        return logits, view1, view2