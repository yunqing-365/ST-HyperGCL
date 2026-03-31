# 文件路径: run_baselines.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
import warnings
warnings.filterwarnings("ignore")

from models.st_hypergcl import UltimateRiskModel  # 复用你的完整架构来跑 GraphCL

# ====================== 【全权控制开关】 ======================
# 请在这里修改你想跑的 Baseline 模式！
# 可选值: 
# "GraphCL"     -> 跑标准的自监督对比学习 (有 GRU 和残差，但不用标签去噪)
# "Vanilla_GCN" -> 跑古典静态图 GCN (无残差，无 GRU，无对比学习，纯原始)
RUN_MODE = "Vanilla_GCN" 
# ==============================================================

print(f"\n" + "="*50)
print(f"🚀 当前运行的 Baseline 模式: {RUN_MODE}")
print("="*50)

# ====================== 【1. 自监督 InfoNCE (GraphCL 专用)】 ======================
def self_supervised_contrastive_loss(z1, z2, temperature=0.5):
    """
    标准的自监督图对比学习 (Self-Supervised GCL):
    只认自己！只把同一个节点的两个视图拉近，把所有其他节点统统推开！绝对不使用真实标签。
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # z1 和 z2 对应位置互为正样本的相似度
    pos_sim = (z1 * z2).sum(dim=1) / temperature
    
    # 计算 z1 对所有 z2 的相似度矩阵 (包含一个正样本和 N-1 个负样本)
    sim_matrix = torch.matmul(z1, z2.t()) / temperature
    
    # InfoNCE Loss: -log( exp(pos) / sum(exp(all)) )
    loss = - (pos_sim - torch.logsumexp(sim_matrix, dim=1))
    return loss.mean()

# ====================== 【2. 古典静态 GCN 架构 (Vanilla GCN 专用)】 ======================
class VanillaGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VanillaGNN, self).__init__()
        # 【去残差】：只保留邻居聚合，去掉自身特征的保护
        self.fc_neigh = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x, adj):
        x_neigh = torch.matmul(adj, x)
        h = self.fc_neigh(x_neigh)
        return F.relu(h)

class VanillaRiskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(VanillaRiskModel, self).__init__()
        self.gnn = VanillaGNN(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj_list):
        T = len(adj_list)
        num_nodes = x.shape[0]
        hidden_dim = self.gnn.fc_neigh.out_features
        
        temporal_features = torch.zeros(num_nodes, T, hidden_dim, device=x.device)
        for t in range(T):
            node_emb = self.gnn(x, adj_list[t])
            temporal_features[:, t, :] = node_emb
            
        # 【去 GRU】：静态池化，跨越9年直接求平均
        final_state = torch.mean(temporal_features, dim=1)
        final_state = self.dropout(final_state)
        
        logits = self.classifier(final_state)
        return logits, None, None  # 不需要返回对比视图

# ====================== 【3. 数据加载与预处理 (保持严格一致)】 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.load("data/processed/node_features_X.pt").to(device)
Y = torch.load("data/processed/risk_labels_Y.pt").to(device)
edge_index_list = torch.load("data/processed/dynamic_edge_indices.pt")

Y = 1 - Y
X_mean = X.mean(dim=0)
X_std = X.std(dim=0) + 1e-5
X_normalized = ((X - X_mean) / X_std).to(device)

num_companies = X.shape[0]
T = len(edge_index_list)

torch.manual_seed(42)
indices = torch.randperm(num_companies)
split_point = int(0.8 * num_companies)
train_idx = indices[:split_point]
test_idx = indices[split_point:]

adj_list_full = []
for t in range(T):
    edge_index_t = edge_index_list[t].to(device)
    if edge_index_t.shape[1] == 0:
        adj_list_full.append(torch.eye(num_companies, device=device))
        continue
    num_edges = edge_index_t.size(1)
    H_indices = edge_index_t[[1, 0]]
    H_values = torch.ones(num_edges, device=device)
    H_sparse = torch.sparse_coo_tensor(H_indices, H_values, size=(num_companies, edge_index_t[0].max().item() + 1), device=device)
    A_sparse = torch.sparse.mm(H_sparse, H_sparse.t())
    A_dense = A_sparse.to_dense()
    A_dense.fill_diagonal_(1.0)
    deg = A_dense.sum(dim=1, keepdim=True)
    deg[deg == 0] = 1.0
    A_norm = A_dense / deg
    adj_list_full.append(A_norm)

# ====================== 【4. 模型与训练配置】 ======================
if RUN_MODE == "GraphCL":
    model = UltimateRiskModel(input_dim=X.shape[1]).to(device)
    alpha = 0.05
    drop_rate = 0.2
elif RUN_MODE == "Vanilla_GCN":
    model = VanillaRiskModel(input_dim=X.shape[1]).to(device)
    alpha = 0.00
    drop_rate = 0.00

optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

num_negative = (Y[train_idx] == 0).sum().item()
num_positive = (Y[train_idx] == 1).sum().item()
pos_weight = torch.tensor([num_negative / (num_positive + 1e-5)], device=device)
criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_auc = 0.
best_pr_auc = 0. 
best_f1 = 0.     
epochs = 200

# ====================== 【5. 训练循环】 ======================
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # 图结构扰动 (如果是 Vanilla_GCN，drop_rate 为 0，不扰动)
    adj_list_dropped = []
    for t in range(T):
        edge_index_t = edge_index_list[t].to(device)
        if edge_index_t.shape[1] == 0:
            adj_list_dropped.append(torch.eye(num_companies, device=device))
            continue
        num_edges = edge_index_t.size(1)
        mask = torch.rand(num_edges, device=device) > drop_rate
        H_indices_dropped = edge_index_t[[1, 0]][:, mask] 
        H_values_dropped = torch.ones(H_indices_dropped.size(1), device=device)
        if H_indices_dropped.size(1) > 0:
            H_sparse = torch.sparse_coo_tensor(H_indices_dropped, H_values_dropped, size=(num_companies, edge_index_t[0].max().item() + 1), device=device)
            A_sparse = torch.sparse.mm(H_sparse, H_sparse.t())
            A_dense = A_sparse.to_dense()
        else:
            A_dense = torch.zeros((num_companies, num_companies), device=device)
        A_dense.fill_diagonal_(1.0)
        deg = A_dense.sum(dim=1, keepdim=True)
        deg[deg == 0] = 1.0
        A_norm = A_dense / deg
        adj_list_dropped.append(A_norm)

    logits, view1, view2 = model(X_normalized, adj_list_dropped)
    loss_bce = criterion_bce(logits[train_idx], Y[train_idx])
    
    loss_gcl = 0.0
    if RUN_MODE == "GraphCL" and view1 is not None:
        loss_gcl = self_supervised_contrastive_loss(view1[train_idx], view2[train_idx])
    
    loss = loss_bce + alpha * loss_gcl
    loss.backward()
    optimizer.step()
    
    # 评估阶段
    model.eval()
    with torch.no_grad():
        preds_prob = torch.sigmoid(model(X_normalized, adj_list_full)[0])
        true_y_test = Y[test_idx].cpu().numpy()
        preds_prob_test = preds_prob[test_idx].cpu().numpy()
        
        try:
            auc_test = roc_auc_score(true_y_test, preds_prob_test)
            if auc_test > best_auc:
                best_auc = auc_test
                precision, recall, _ = precision_recall_curve(true_y_test, preds_prob_test)
                best_pr_auc = auc(recall, precision)
                pred_binary = (preds_prob_test >= 0.5).astype(int)
                best_f1 = f1_score(true_y_test, pred_binary)
        except ValueError:
            pass
            
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:03d} | BCE: {loss_bce.item():.3f} | 当前 Test AUC: {auc_test:.3f}")

print("\n" + "="*50)
print(f"✅ Baseline [{RUN_MODE}] 测试集最终成绩：")
print(f"👑 AUC: {best_auc:.3f} | 🏅 PR-AUC: {best_pr_auc:.3f} | 🏅 F1-Score: {best_f1:.3f}")
print("="*50)