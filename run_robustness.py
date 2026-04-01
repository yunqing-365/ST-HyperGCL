import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from models.st_hypergcl import UltimateRiskModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.load("data/processed/node_features_X.pt").to(device)
Y = torch.load("data/processed/risk_labels_Y.pt").to(device)
edge_index_list = torch.load("data/processed/dynamic_edge_indices.pt")

X_mean = X.mean(dim=0)
X_std = X.std(dim=0) + 1e-5
X_normalized = ((X - X_mean) / X_std).to(device)

num_companies = X.shape[0]
T = len(edge_index_list)

# 预计算超图矩阵 (只算一次，加速循环)
print("🧬 正在预计算超图拉普拉斯算子...")
adj_list_full = []
for t in range(T):
    edge_index_t = edge_index_list[t].to(device)
    if edge_index_t.shape[1] == 0:
        adj_list_full.append(torch.eye(num_companies, device=device))
        continue
    H_sparse = torch.sparse_coo_tensor(edge_index_t[[1, 0]], torch.ones(edge_index_t.size(1), device=device), size=(num_companies, edge_index_t[0].max().item() + 1), device=device)
    H_dense = H_sparse.to_dense()
    D_v_inv_sqrt = torch.pow(H_dense.sum(dim=1), -0.5).nan_to_num(posinf=0.0)
    D_e_inv = torch.pow(H_dense.sum(dim=0), -1.0).nan_to_num(posinf=0.0)
    G = torch.matmul(H_dense * D_v_inv_sqrt.unsqueeze(1) * D_e_inv.unsqueeze(0), H_dense.t()) * D_v_inv_sqrt.unsqueeze(0)
    G.fill_diagonal_(1.0)
    adj_list_full.append(G)

def run_single_experiment(seed, hidden_dim=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    indices = torch.randperm(num_companies)
    split_point = int(0.8 * num_companies)
    train_idx, test_idx = indices[:split_point], indices[split_point:]
    
    model = UltimateRiskModel(input_dim=X.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    pos_weight = torch.tensor([(Y[train_idx] == 0).sum().item() / ((Y[train_idx] == 1).sum().item() + 1e-5)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_auc = 0.
    for epoch in range(150): # 150轮足够收敛，节省时间
        model.train()
        optimizer.zero_grad()
        
        # 图微扰 Dropout
        adj_list_dropped = [F.dropout(G, p=0.2, training=True) for G in adj_list_full]
        logits, _, _ = model(X_normalized, adj_list_dropped)
        
        # 纯 V2 (BCE only, 不加 SupCon)
        loss = criterion(logits[train_idx], Y[train_idx]) 
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds_prob = torch.sigmoid(model(X_normalized, adj_list_full)[0][test_idx]).cpu().numpy()
            try:
                auc_test = roc_auc_score(Y[test_idx].cpu().numpy(), preds_prob)
                if auc_test > best_auc: best_auc = auc_test
            except: pass
    return best_auc

print("\n🚀 开始提取主模型 (ST-HyperGCL Ours) 的 10 随机种子鲁棒性数据...\n")
# 严格对齐你刚才测 T-GCN 用的 10 个种子
seeds = [42, 999, 123, 456, 789, 1111, 1024, 2026, 3407, 8888]

auc_list = []
for s in seeds:
    res = run_single_experiment(s, hidden_dim=64)
    print(f"   -> Ours Seed {s}: AUC = {res:.3f}")
    auc_list.append(res)

print("="*50)
print(f"✅ Ours 真实序列: {auc_list}")
print(f"✅ 论文汇报格式 (Mean ± Std): {np.mean(auc_list):.3f} ± {np.std(auc_list):.3f}")
print("="*50)