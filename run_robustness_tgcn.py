import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.load("data/processed/node_features_X.pt").to(device)
Y = torch.load("data/processed/risk_labels_Y.pt").to(device)
edge_index_list = torch.load("data/processed/dynamic_edge_indices.pt")

X_mean = X.mean(dim=0)
X_std = X.std(dim=0) + 1e-5
X_normalized = ((X - X_mean) / X_std).to(device)

num_companies = X.shape[0]
T = len(edge_index_list)

class TGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.gcn1 = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, adjs):
        temporal_states = []
        for adj in adjs:
            h = torch.matmul(adj, x)
            h = F.relu(self.gcn1(h))
            temporal_states.append(h.unsqueeze(1))
        out_seq = torch.cat(temporal_states, dim=1)
        gru_out, _ = self.gru(out_seq)
        logits = self.classifier(gru_out[:, -1, :])
        return logits

def run_tgcn_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    indices = torch.randperm(num_companies)
    split_point = int(0.8 * num_companies)
    train_idx, test_idx = indices[:split_point], indices[split_point:]
    
    # 动态普通图归一化
    adj_list = []
    for t in range(T):
        edge_index_t = edge_index_list[t].to(device)
        if edge_index_t.shape[1] > 0:
            H_sparse = torch.sparse_coo_tensor(edge_index_t[[1,0]], torch.ones(edge_index_t.size(1), device=device), size=(num_companies, edge_index_t[0].max().item() + 1))
            H_dense = H_sparse.to_dense()
            A = (torch.matmul(H_dense, H_dense.t()) > 0).float()
        else:
            A = torch.eye(num_companies, device=device)
        A.fill_diagonal_(1.0)
        D_inv_sqrt = torch.pow(A.sum(dim=1), -0.5).nan_to_num(posinf=0.0)
        G_norm = A * D_inv_sqrt.unsqueeze(1) * D_inv_sqrt.unsqueeze(0)
        adj_list.append(G_norm)
        
    model = TGCN(in_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    pos_weight = torch.tensor([(Y[train_idx] == 0).sum().item() / ((Y[train_idx] == 1).sum().item() + 1e-5)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_auc = 0.
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_normalized, adj_list)[train_idx], Y[train_idx])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(X_normalized, adj_list)[test_idx]).cpu().numpy()
            try:
                auc_test = roc_auc_score(Y[test_idx].cpu().numpy(), preds)
                if auc_test > best_auc: best_auc = auc_test
            except: pass
    return best_auc

print("\n🚀 正在提取真实 T-GCN 基线的多随机种子鲁棒性数据...\n")
seeds = [42, 999,123,456,789,1111,1024, 2026, 3407, 8888]
tgcn_aucs = []
for s in seeds:
    res = run_tgcn_seed(s)
    print(f"   -> T-GCN Seed {s}: AUC = {res:.3f}")
    tgcn_aucs.append(res)
print(f"✅ T-GCN 真实序列: {tgcn_aucs}")