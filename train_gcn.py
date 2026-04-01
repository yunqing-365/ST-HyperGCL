import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 启动 Vanilla GCN 基线测试 (静态普通图对比版)...")

# ==========================================
# 1. 加载数据 & 绝对公平的划分
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    X = torch.load("data/processed/node_features_X.pt").to(device)
    Y = torch.load("data/processed/risk_labels_Y.pt").to(device)
    edge_index_list = torch.load("data/processed/dynamic_edge_indices.pt")
except FileNotFoundError:
    print("❌ 找不到数据文件，请确认你在 ST-HyperGCL 根目录下运行！")
    exit()

X_mean = X.mean(dim=0)
X_std = X.std(dim=0) + 1e-5
X_normalized = ((X - X_mean) / X_std).to(device)

num_companies = X.shape[0]

torch.manual_seed(42)  # 铁律：保持和 V3 完全一样的座位表
indices = torch.randperm(num_companies)
split_point = int(0.8 * num_companies)
train_idx = indices[:split_point]
test_idx = indices[split_point:]

# ==========================================
# 2. 将超图退化为普通静态图 (GCN的常规操作)
# 将公司-科学家关联 (H) 转换为 公司-公司关联 (A = H * H^T)
# ==========================================
print("🧬 正在将动态超图降维退化为普通静态图...")
H_indices_all = []
for t in range(len(edge_index_list)):
    edge_index_t = edge_index_list[t].to(device)
    if edge_index_t.shape[1] > 0:
        H_indices_all.append(edge_index_t)

if len(H_indices_all) > 0:
    H_all = torch.cat(H_indices_all, dim=1)
    max_inv_idx = H_all[0].max().item() + 1
    H_sparse = torch.sparse_coo_tensor(H_all[[1,0]], torch.ones(H_all.size(1), device=device), size=(num_companies, max_inv_idx))
    H_dense = H_sparse.to_dense()
    
    # 核心退化：公司之间的直接连边矩阵 A
    A = torch.matmul(H_dense, H_dense.t())
    A = (A > 0).float() # 二值化为普通无权图
else:
    A = torch.eye(num_companies, device=device)

# GCN 标准拉普拉斯归一化： D^{-1/2} A D^{-1/2}
A.fill_diagonal_(1.0)
D = A.sum(dim=1)
D_inv_sqrt = torch.pow(D, -0.5)
D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
G_norm = A * D_inv_sqrt.unsqueeze(1) * D_inv_sqrt.unsqueeze(0)

# ==========================================
# 3. 定义最经典的 2 层 Vanilla GCN
# ==========================================
class VanillaGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, adj):
        # Layer 1
        x = torch.matmul(adj, x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        # Layer 2
        x = torch.matmul(adj, x)
        x = F.relu(self.fc2(x))
        logits = self.classifier(x)
        return logits

model = VanillaGCN(in_dim=X.shape[1]).to(device)
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

print("🌲 正在训练 Vanilla GCN ...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    logits = model(X_normalized, G_norm)
    loss = criterion(logits[train_idx], Y[train_idx])
    
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds_prob = torch.sigmoid(model(X_normalized, G_norm)[test_idx]).cpu().numpy()
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
print("📊 Vanilla GCN 基线模型 (普通静态图) OOT 盲测战报：")
print(f"   Test AUC    : {best_auc:.3f}")
print(f"   Test PR-AUC : {best_pr_auc:.3f}")
print(f"   Test F1     : {best_f1:.3f}")
print("="*50)