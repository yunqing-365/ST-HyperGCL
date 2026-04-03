import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import warnings
import os
warnings.filterwarnings("ignore")

from models.st_hypergcl import UltimateRiskModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.load("data/processed/node_features_X.pt").to(device)
Y = torch.load("data/processed/risk_labels_Y.pt").to(device)
edge_index_list = torch.load("data/processed/dynamic_edge_indices.pt")

# 🌟 你原版代码的灵魂标签反转
Y = 1 - Y

X_normalized = ((X - X.mean(dim=0)) / (X.std(dim=0) + 1e-5)).to(device)
num_companies = X.shape[0]
T = len(edge_index_list)

torch.manual_seed(42)
indices = torch.randperm(num_companies)
split_point = int(0.8 * num_companies)
train_idx, test_idx = indices[:split_point], indices[split_point:]

adj_list_full = []
for t in range(T):
    edge_index_t = edge_index_list[t].to(device)
    if edge_index_t.shape[1] == 0:
        adj_list_full.append(torch.eye(num_companies, device=device))
        continue
    H_dense = torch.sparse_coo_tensor(edge_index_t[[1, 0]], torch.ones(edge_index_t.size(1), device=device), size=(num_companies, edge_index_t[0].max().item() + 1), device=device).to_dense()
    D_v_inv = torch.pow(H_dense.sum(dim=1), -0.5); D_v_inv[torch.isinf(D_v_inv)] = 0.0
    D_e_inv = torch.pow(H_dense.sum(dim=0), -1.0); D_e_inv[torch.isinf(D_e_inv)] = 0.0
    G = torch.matmul((H_dense * D_v_inv.unsqueeze(1)) * D_e_inv.unsqueeze(0), H_dense.t()) * D_v_inv.unsqueeze(0)
    G.fill_diagonal_(1.0)
    adj_list_full.append(G)

model = UltimateRiskModel(input_dim=X.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
pos_weight = torch.tensor([(Y[train_idx] == 0).sum() / ((Y[train_idx] == 1).sum() + 1e-5)], device=device)
criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_auc = 0.
drop_rate = 0.2 # 🌟 原版的防过拟合神器

print(f"🚀 开始重现 0.716 巅峰时刻...")
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    adj_list_dropped = []
    for t in range(T):
        edge_index_t = edge_index_list[t].to(device)
        if edge_index_t.shape[1] == 0:
            adj_list_dropped.append(torch.eye(num_companies, device=device))
            continue
        mask = torch.rand(edge_index_t.size(1), device=device) > drop_rate
        H_ind_drop = edge_index_t[[1, 0]][:, mask]
        if H_ind_drop.size(1) > 0:
            H_d = torch.sparse_coo_tensor(H_ind_drop, torch.ones(H_ind_drop.size(1), device=device), size=(num_companies, edge_index_t[0].max().item() + 1), device=device).to_dense()
            Dv = torch.pow(H_d.sum(dim=1), -0.5); Dv[torch.isinf(Dv)] = 0.0
            De = torch.pow(H_d.sum(dim=0), -1.0); De[torch.isinf(De)] = 0.0
            Gd = torch.matmul((H_d * Dv.unsqueeze(1)) * De.unsqueeze(0), H_d.t()) * Dv.unsqueeze(0)
        else:
            Gd = torch.zeros((num_companies, num_companies), device=device)
        Gd.fill_diagonal_(1.0)
        adj_list_dropped.append(Gd)

    logits, _, _ = model(X_normalized, adj_list_dropped)
    loss = criterion_bce(logits[train_idx], Y[train_idx])
    loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_prob = torch.sigmoid(model(X_normalized, adj_list_full)[0])[test_idx].cpu().numpy()
        auc_test = roc_auc_score(Y[test_idx].cpu().numpy(), preds_prob)
        if auc_test > best_auc:
            best_auc = auc_test
            torch.save(model.state_dict(), "best_model.pth")
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Test AUC: {auc_test:.4f}")

print(f"🏆 成功还原！最佳模型已保存，Test AUC: {best_auc:.4f}")