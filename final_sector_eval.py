import torch
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from models.st_hypergcl import UltimateRiskModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processed_dir = "data/processed/"
X = torch.load(os.path.join(processed_dir, "node_features_X.pt")).to(device)
Y = torch.load(os.path.join(processed_dir, "risk_labels_Y.pt")).to(device)
Y = 1 - Y # 🌟 必须匹配训练时的反转
edge_index_list = torch.load(os.path.join(processed_dir, "dynamic_edge_indices.pt"))
stkcd_order = torch.load(os.path.join(processed_dir, "stkcd_order.pt"))

X_normalized = ((X - X.mean(dim=0)) / (X.std(dim=0) + 1e-5)).to(device)
num_companies = X.shape[0]

torch.manual_seed(42)
test_idx = torch.randperm(num_companies)[int(0.8 * num_companies):].numpy()

# 生成行业掩码
ind_df = pd.read_csv("data/raw_data/STK_INDUSTRYCLASS.csv", encoding='utf-8')
ind_df['Symbol'] = ind_df['Symbol'].astype(str).str.zfill(6)
ind_df = ind_df[ind_df['IndustryClassificationID'] == 'P0207']
ind_df = ind_df.sort_values('ImplementDate').drop_duplicates('Symbol', keep='last')
industry_map = dict(zip(ind_df['Symbol'], ind_df['IndustryCode']))

formatted_order = [str(x).split('.')[0].zfill(6) for x in stkcd_order]
c38_mask = np.array([industry_map.get(s) == 'C38' for s in formatted_order])
c39_mask = np.array([industry_map.get(s) == 'C39' for s in formatted_order])

# 图结构
adj_list_full = []
for t in range(len(edge_index_list)):
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
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

with torch.no_grad():
    probs = torch.sigmoid(model(X_normalized, adj_list_full)[0]).cpu().numpy()
    y_true = Y.cpu().numpy()

def eval_sector(sector_mask, sector_name):
    target_idx = [i for i in test_idx if sector_mask[i]]
    y_s, p_s = y_true[target_idx], probs[target_idx]
    precision, recall, _ = precision_recall_curve(y_s, p_s)
    print(f"🎯 {sector_name:15} | 样本量: {len(target_idx):3} | AUC: {roc_auc_score(y_s, p_s):.4f} | PR-AUC: {auc(recall, precision):.4f}")

print("\n" + "="*60)
eval_sector(c38_mask, "C38 (电气机械)")
eval_sector(c39_mask, "C39 (电子设备)")
print("="*60)