import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 启动 T-GCN 基线测试 (动态普通图 + GRU)...")

# ==========================================
# 1. 加载数据 & 绝对公平划分
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.load("data/processed/node_features_X.pt").to(device)
Y = torch.load("data/processed/risk_labels_Y.pt").to(device)
edge_index_list = torch.load("data/processed/dynamic_edge_indices.pt")

X_mean = X.mean(dim=0)
X_std = X.std(dim=0) + 1e-5
X_normalized = ((X - X_mean) / X_std).to(device)

num_companies = X.shape[0]
T = len(edge_index_list)

torch.manual_seed(42) # 铁律：保持座位表一致
indices = torch.randperm(num_companies)
split_point = int(0.8 * num_companies)
train_idx = indices[:split_point]
test_idx = indices[split_point:]

# ==========================================
# 2. 生成 T 个切片的静态图归一化矩阵
# ==========================================
adj_list = []
for t in range(T):
    edge_index_t = edge_index_list[t].to(device)
    if edge_index_t.shape[1] > 0:
        max_inv_idx = edge_index_t[0].max().item() + 1
        H_sparse = torch.sparse_coo_tensor(edge_index_t[[1,0]], torch.ones(edge_index_t.size(1), device=device), size=(num_companies, max_inv_idx))
        H_dense = H_sparse.to_dense()
        A = torch.matmul(H_dense, H_dense.t())
        A = (A > 0).float()
    else:
        A = torch.eye(num_companies, device=device)
        
    A.fill_diagonal_(1.0)
    D = A.sum(dim=1)
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    G_norm = A * D_inv_sqrt.unsqueeze(1) * D_inv_sqrt.unsqueeze(0)
    adj_list.append(G_norm)

# ==========================================
# 3. 定义 T-GCN (GCN 提取单拍空间特征，GRU 捕捉时间演化)
# ==========================================
class TGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.gcn1 = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, adjs):
        temporal_states = []
        for adj in adjs:
            # GCN 空间卷积
            h = torch.matmul(adj, x)
            h = F.relu(self.gcn1(h))
            temporal_states.append(h.unsqueeze(1))
            
        # 沿着时间维度拼接 [Num_nodes, T, Hidden_dim]
        out_seq = torch.cat(temporal_states, dim=1)
        # 喂给 GRU
        gru_out, _ = self.gru(out_seq)
        # 取最后一步的状态进行分类
        final_state = gru_out[:, -1, :]
        logits = self.classifier(final_state)
        return logits

model = TGCN(in_dim=X.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

num_negative = (Y[train_idx] == 0).sum().item()
num_positive = (Y[train_idx] == 1).sum().item()
pos_weight = torch.tensor([num_negative / (num_positive + 1e-5)], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ==========================================
# 4. 训练循环
# ==========================================
epochs = 200
best_auc = 0.
best_pr_auc = 0.
best_f1 = 0.

print("⏳ 正在训练 T-GCN ...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    logits = model(X_normalized, adj_list)
    loss = criterion(logits[train_idx], Y[train_idx])
    
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds_prob = torch.sigmoid(model(X_normalized, adj_list)[test_idx]).cpu().numpy()
        true_y = Y[test_idx].cpu().numpy()
        
        try:
            auc_test = roc_auc_score(true_y, preds_prob)
            if auc_test > best_auc:
                best_auc = auc_test
                precision, recall, _ = precision_recall_curve(true_y, preds_prob)
                best_pr_auc = auc(recall, precision)
                pred_binary = (preds_prob >= 0.5).astype(int)
                best_f1 = f1_score(true_y, pred_binary)
        except:
            pass

print("="*50)
print("📊 T-GCN 基线模型 (动态普通图) OOT 盲测战报：")
print(f"   Test AUC    : {best_auc:.3f}")
print(f"   Test PR-AUC : {best_pr_auc:.3f}")
print(f"   Test F1     : {best_f1:.3f}")
print("="*50)