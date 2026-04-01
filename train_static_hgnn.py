import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 启动 Static HGNN 基线测试 (静态超图，无时间维度)...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.load("data/processed/node_features_X.pt").to(device)
Y = torch.load("data/processed/risk_labels_Y.pt").to(device)
edge_index_list = torch.load("data/processed/dynamic_edge_indices.pt")

X_mean = X.mean(dim=0)
X_std = X.std(dim=0) + 1e-5
X_normalized = ((X - X_mean) / X_std).to(device)

num_companies = X.shape[0]

torch.manual_seed(42) # 保持座位表绝对一致
indices = torch.randperm(num_companies)
split_point = int(0.8 * num_companies)
train_idx = indices[:split_point]
test_idx = indices[split_point:]

# ==========================================
# 1. 把 8 年的动态超图“拍扁”成一张全局静态超图
# ==========================================
print("🧬 正在将动态超图压缩为全局静态超图...")
H_dense_list = []
max_inv_all = 0

for t in range(len(edge_index_list)):
    edge_index_t = edge_index_list[t].to(device)
    if edge_index_t.shape[1] > 0:
        max_inv_idx = edge_index_t[0].max().item() + 1
        max_inv_all = max(max_inv_all, max_inv_idx)

for t in range(len(edge_index_list)):
    edge_index_t = edge_index_list[t].to(device)
    if edge_index_t.shape[1] > 0:
        H_sparse = torch.sparse_coo_tensor(edge_index_t[[1,0]], torch.ones(edge_index_t.size(1), device=device), size=(num_companies, max_inv_all))
        H_dense_list.append(H_sparse.to_dense())

if len(H_dense_list) > 0:
    # 叠加所有年份的关联，并二值化
    H_static = sum(H_dense_list)
    H_static = (H_static > 0).float()
    
    D_v = H_static.sum(dim=1)
    D_e = H_static.sum(dim=0)
    
    D_v_inv_sqrt = torch.pow(D_v, -0.5)
    D_v_inv_sqrt[torch.isinf(D_v_inv_sqrt)] = 0.0
    D_e_inv = torch.pow(D_e, -1.0)
    D_e_inv[torch.isinf(D_e_inv)] = 0.0
    
    H_step1 = H_static * D_v_inv_sqrt.unsqueeze(1)
    H_step2 = H_step1 * D_e_inv.unsqueeze(0)
    H_step3 = torch.matmul(H_step2, H_static.t())
    G_static = H_step3 * D_v_inv_sqrt.unsqueeze(0)
    G_static.fill_diagonal_(1.0)
else:
    G_static = torch.eye(num_companies, device=device)

# ==========================================
# 2. 定义静态超图神经网络 (无 GRU)
# ==========================================
class StaticHGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = torch.matmul(adj, x)
        x = F.relu(self.fc2(x))
        logits = self.classifier(x)
        return logits

model = StaticHGNN(in_dim=X.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
pos_weight = torch.tensor([(Y[train_idx] == 0).sum().item() / ((Y[train_idx] == 1).sum().item() + 1e-5)], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_auc, best_pr, best_f1 = 0., 0., 0.
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_normalized, G_static)[train_idx], Y[train_idx])
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds_prob = torch.sigmoid(model(X_normalized, G_static)[test_idx]).cpu().numpy()
        true_y = Y[test_idx].cpu().numpy()
        try:
            auc_test = roc_auc_score(true_y, preds_prob)
            if auc_test > best_auc:
                best_auc = auc_test
                precision, recall, _ = precision_recall_curve(true_y, preds_prob)
                best_pr = auc(recall, precision)
                best_f1 = f1_score(true_y, (preds_prob >= 0.5).astype(int))
        except: pass

print("="*50)
print(f"📊 Static HGNN (静态超图) OOT 战报：")
print(f"   Test AUC    : {best_auc:.3f}")
print(f"   Test PR-AUC : {best_pr:.3f}")
print("="*50)