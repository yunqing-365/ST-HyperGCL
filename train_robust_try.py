import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

from models.st_hypergcl import UltimateRiskModel, contrastive_loss

# ====================== 【关键修复 1：自动使用 GPU】======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

print("加载时序动态张量数据")

# 1. 加载数据 + 直接放到 GPU
X = torch.load("data/processed/node_features_X.pt").to(device)
Y = torch.load("data/processed/risk_labels_Y.pt").to(device)
edge_index_list = torch.load("data/processed/dynamic_edge_indices.pt")

# 标准化
X_mean = X.mean(dim=0)
X_std = X.std(dim=0) + 1e-5
X_normalized = ((X - X_mean) / X_std).to(device)

num_companies = X.shape[0]
T = len(edge_index_list)

# ====================== 【关键修复 2：轻量化邻接矩阵，永不卡死】======================
adj_list = []
for t in range(T):
    edge_index_t = edge_index_list[t].to(device)
    if edge_index_t.shape[1] == 0:
        adj_list.append(torch.eye(num_companies, device=device))
        continue

    # ---------------- 轻量化超图计算（不生成巨型矩阵）----------------
    num_edges = edge_index_t.size(1)
    H_indices = edge_index_t[[1, 0]]  # [company, inventor]
    H_values = torch.ones(num_edges, device=device)

    # 稀疏矩阵，不占内存
    H_sparse = torch.sparse_coo_tensor(
        H_indices, H_values,
        size=(num_companies, edge_index_t[0].max().item() + 1),
        device=device
    )

    # 轻量化 A 矩阵
    A_sparse = torch.sparse.mm(H_sparse, H_sparse.t())
    A_dense = A_sparse.to_dense()
    A_dense.fill_diagonal_(1.0)

    deg = A_dense.sum(dim=1, keepdim=True)
    deg[deg == 0] = 1.0
    A_norm = A_dense / deg
    adj_list.append(A_norm)

# 3. 模型放到 GPU
model = UltimateRiskModel(input_dim=X.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
pos_weight = torch.tensor([5.0], device=device)
criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 4. 训练
epochs = 200
alpha = 0.5

print("\n🚀 开始【10% 高斯噪声】抗打击测试...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # 加噪声
    noise_level = 0.10
    noise = torch.randn_like(X_normalized) * noise_level
    noisy_X = X_normalized + noise

    # 前向
    logits, view1, view2 = model(noisy_X, adj_list)

    # 损失
    loss_bce = criterion_bce(logits, Y)
    loss_gcl = contrastive_loss(view1, view2)
    loss = loss_bce + alpha * loss_gcl

    loss.backward()
    optimizer.step()

    # 评估
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_noise = torch.randn_like(X_normalized) * noise_level
            test_noisy_X = X_normalized + test_noise
            preds_prob = torch.sigmoid(model(test_noisy_X, adj_list)[0])

            true_y = Y.cpu().numpy()
            preds_prob = preds_prob.cpu().numpy()
            preds_label = (preds_prob > 0.5).astype(float)

            acc = accuracy_score(true_y, preds_label)
            try:
                auc = roc_auc_score(true_y, preds_prob)
            except ValueError:
                auc = 0.5

            print(f"Epoch {epoch+1}/{epochs} | BCE: {loss_bce.item():.3f} | GCL: {loss_gcl.item():.3f} | 抗噪 AUC: {auc:.3f}")

print("\n🏆 运行完成！")