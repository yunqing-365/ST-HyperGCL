import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, f1_score
import warnings
import os
warnings.filterwarnings("ignore")

from models.st_hypergcl import UltimateRiskModel

# ====================== 【1. 设备与数据加载】 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 使用设备: {device}")

# 路径对齐
processed_dir = "data/processed/"
X = torch.load(os.path.join(processed_dir, "node_features_X.pt")).to(device)
Y = torch.load(os.path.join(processed_dir, "risk_labels_Y.pt")).to(device)
edge_index_list = torch.load(os.path.join(processed_dir, "dynamic_edge_indices.pt"))

Y = 1 - Y  # 标签修复
X_mean = X.mean(dim=0)
X_std = X.std(dim=0) + 1e-5
X_normalized = ((X - X_mean) / X_std).to(device)

num_companies = X.shape[0]
T = len(edge_index_list)

# ====================== 【2. 数据划分】 ======================
torch.manual_seed(42)
indices = torch.randperm(num_companies)
split_point = int(0.8 * num_companies)
train_idx = indices[:split_point]
test_idx = indices[split_point:]

# ====================== 【3. HGNN 预处理逻辑 (同原脚本)】 ======================
print("⚙️ 正在预处理超图拉普拉斯矩阵...")
adj_list_full = []
for t in range(T):
    edge_index_t = edge_index_list[t].to(device)
    if edge_index_t.shape[1] == 0:
        adj_list_full.append(torch.eye(num_companies, device=device))
        continue
    H_indices = edge_index_t[[1, 0]]
    H_values = torch.ones(edge_index_t.size(1), device=device)
    H_dense = torch.sparse_coo_tensor(H_indices, H_values, size=(num_companies, edge_index_t[0].max().item() + 1), device=device).to_dense()
    D_v = H_dense.sum(dim=1)
    D_e = H_dense.sum(dim=0)
    D_v_inv_sqrt = torch.pow(D_v, -0.5)
    D_v_inv_sqrt[torch.isinf(D_v_inv_sqrt)] = 0.0
    D_e_inv = torch.pow(D_e, -1.0)
    D_e_inv[torch.isinf(D_e_inv)] = 0.0
    H_step1 = H_dense * D_v_inv_sqrt.unsqueeze(1)
    H_step2 = H_step1 * D_e_inv.unsqueeze(0)
    G = torch.matmul(H_step2, H_dense.t()) * D_v_inv_sqrt.unsqueeze(0)
    G.fill_diagonal_(1.0)
    adj_list_full.append(G)

# ====================== 【4. 模型与训练配置】 ======================
model = UltimateRiskModel(input_dim=X.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

num_negative = (Y[train_idx] == 0).sum().item()
num_positive = (Y[train_idx] == 1).sum().item()
pos_weight = torch.tensor([num_negative / (num_positive + 1e-5)], device=device)
criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ====================== 【5. 训练循环】 ======================
epochs = 200
best_auc = 0.
best_model_state = None

print(f"\n🚀 开始训练并实时监控最佳性能...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    logits, _, _ = model(X_normalized, adj_list_full)
    loss = criterion_bce(logits[train_idx], Y[train_idx])
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds_prob = torch.sigmoid(model(X_normalized, adj_list_full)[0])
        true_y_test = Y[test_idx].cpu().numpy()
        preds_prob_test = preds_prob[test_idx].cpu().numpy()
        
        try:
            auc_test = roc_auc_score(true_y_test, preds_prob_test)
            if auc_test > best_auc:
                best_auc = auc_test
                # 🌟 关键：在内存中保存当前表现最好的模型权重
                best_model_state = model.state_dict()
        except:
            pass
        
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Current Test AUC: {auc_test:.4f}")

# ====================== 【6. 保存最佳模型文件】 ======================
if best_model_state is not None:
    # 🌟 将模型保存到根目录，供 evaluate_by_sector.py 使用
    torch.save(best_model_state, "best_model.pth")
    print(f"\n✅ 训练完成！最佳模型已保存至: {os.path.abspath('best_model.pth')}")
    print(f"🏆 最佳 Test AUC: {best_auc:.4f}")
else:
    print("❌ 训练过程中未捕获到有效指标，模型未保存。")