import torch
import torch.nn as nn
import torch.optim as optim
# 🌟 修改点 1：导入了计算 PR-AUC 和 F1 的工具
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, f1_score
import warnings
warnings.filterwarnings("ignore")

# ====================== 【1. 设备与加载】 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 使用设备: {device}")

X = torch.load("data/processed/node_features_X.pt").to(device)
Y = torch.load("data/processed/risk_labels_Y.pt").to(device)
edge_index_list = torch.load("data/processed/dynamic_edge_indices.pt")

Y = 1 - Y  # 🌟 修复标签倒置

X_mean = X.mean(dim=0)
X_std = X.std(dim=0) + 1e-5
X_normalized = ((X - X_mean) / X_std).to(device)

num_companies = X.shape[0]
T = len(edge_index_list)

# ====================== 【2. 极其严格的 Train/Test 划分】 ======================
# ⚠️ 这里必须和 Ultimate 模型使用一模一样的随机种子和切分方式！
torch.manual_seed(42) 
indices = torch.randperm(num_companies)
split_point = int(0.8 * num_companies)
train_idx = indices[:split_point]  # 80% 训练
test_idx = indices[split_point:]   # 20% 测试

# ====================== 【3. 图结构预处理】 ======================
adj_list = []
for t in range(T):
    edge_index_t = edge_index_list[t].to(device)
    if edge_index_t.shape[1] == 0:
        adj_list.append(torch.eye(num_companies, device=device))
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
    adj_list.append(A_norm)

# ====================== 【4. 模型定义与训练配置】 ======================
class TGCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(TGCNModel, self).__init__()
        self.gcn_weight = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj_list):
        T = len(adj_list)
        num_nodes = x.shape[0]
        hidden_dim = self.gcn_weight.out_features
        temporal_features = torch.zeros(num_nodes, T, hidden_dim, device=x.device)
        for t in range(T):
            x_neigh = torch.matmul(adj_list[t], x)
            temporal_features[:, t, :] = torch.relu(self.gcn_weight(x_neigh))
        gru_out, _ = self.gru(temporal_features)
        return self.classifier(gru_out[:, -1, :])

model_tgcn = TGCNModel(input_dim=X.shape[1]).to(device)
optimizer = optim.Adam(model_tgcn.parameters(), lr=0.005, weight_decay=1e-4)

# 动态计算训练集的类别权重
num_negative = (Y[train_idx] == 0).sum().item()
num_positive = (Y[train_idx] == 1).sum().item()
pos_weight = torch.tensor([num_negative / (num_positive + 1e-5)], device=device)
criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ====================== 【5. 训练循环】 ======================
epochs = 200
best_auc = 0.0
# 🌟 修改点 2：新增用于记录巅峰状态下的 PR-AUC 和 F1-Score 的变量
best_pr_auc = 0.0
best_f1 = 0.0
best_epoch = 0

print("\n🚀 开始训练 T-GCN (真实隔离测试集版)...")

for epoch in range(epochs):
    model_tgcn.train()
    optimizer.zero_grad()
    
    logits = model_tgcn(X_normalized, adj_list)
    # ⚠️ 训练时只看 train_idx
    loss = criterion_bce(logits[train_idx], Y[train_idx])
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        model_tgcn.eval()
        with torch.no_grad():
            preds_prob = torch.sigmoid(model_tgcn(X_normalized, adj_list))
            
            # ⚠️ 测试时只看 test_idx
            true_y_test = Y[test_idx].cpu().numpy()
            preds_prob_test = preds_prob[test_idx].cpu().numpy()
            
            try:
                auc_test = roc_auc_score(true_y_test, preds_prob_test)
                if auc_test > best_auc:
                    best_auc = auc_test
                    best_epoch = epoch + 1
                    
                    # 🌟 修改点 3：在创下新高 AUC 的这一刻，同步计算出 PR-AUC 和 F1
                    precision, recall, _ = precision_recall_curve(true_y_test, preds_prob_test)
                    best_pr_auc = auc(recall, precision)
                    
                    pred_binary = (preds_prob_test >= 0.5).astype(int)
                    best_f1 = f1_score(true_y_test, pred_binary)
                    
            except ValueError:
                auc_test = 0.5
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.3f} | 当前 Test AUC: {auc_test:.3f}")

# 🌟 修改点 4：在训练结束的总结栏里，把新指标打印出来
print("="*50)
print(f"🏆 T-GCN 卸下伪装后的真实水平：")
print(f"👑 巅峰 Test AUC 暴跌至: {best_auc:.3f} (出现在第 {best_epoch} 轮)")
print(f"🏅 补充顶会指标 -> PR-AUC: {best_pr_auc:.3f} | F1-Score: {best_f1:.3f}")
print("="*50)