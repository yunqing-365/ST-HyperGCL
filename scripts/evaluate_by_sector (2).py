import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

# --- 1. 自动处理项目路径 (解决 ModuleNotFoundError) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在可以安全导入模型了
try:
    from models.st_hypergcl import UltimateRiskModel
except ImportError:
    print("❌ 错误：找不到 models.st_hypergcl 模块。请确保在 ST-HyperGCL 根目录下运行或检查文件夹结构。")
    sys.exit(1)

# --- 2. 配置与路径设定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processed_dir = os.path.join(project_root, "data", "processed")
model_save_path = os.path.join(project_root, "best_model.pth") # 假设模型保存在根目录

print(f"🖥️ 使用设备: {device}")

# --- 3. 加载数据与掩码 ---
# 必须先确保这些文件存在
files_to_check = ["node_features_X.pt", "risk_labels_Y.pt", "dynamic_edge_indices.pt", "sector_masks.pt"]
for f in files_to_check:
    if not os.path.exists(os.path.join(processed_dir, f)):
        print(f"❌ 错误：缺少必要的数据文件 {f}。请确保已运行之前的生成脚本。")
        sys.exit(1)

X = torch.load(os.path.join(processed_dir, "node_features_X.pt")).to(device)
Y = torch.load(os.path.join(processed_dir, "risk_labels_Y.pt")).to(device)
edge_index_list = torch.load(os.path.join(processed_dir, "dynamic_edge_indices.pt"))
masks = torch.load(os.path.join(processed_dir, "sector_masks.pt"))

# 标签修复与特征归一化 (逻辑同 train_v2_true_hgnn.py)
Y = 1 - Y 
X_mean = X.mean(dim=0)
X_std = X.std(dim=0) + 1e-5
X_normalized = ((X - X_mean) / X_std).to(device)

num_companies = X.shape[0]
T = len(edge_index_list)

# --- 4. 复现测试集划分 (Seed 42) ---
torch.manual_seed(42)
indices = torch.randperm(num_companies)
test_idx = indices[int(0.8 * num_companies):]

# --- 5. 准备 HGNN 预处理矩阵 (复现 train_v2 逻辑) ---
print("⚙️ 正在生成超图卷积矩阵...")
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

# --- 6. 加载模型并推理 ---
if not os.path.exists(model_save_path):
    print(f"❌ 错误：找不到训练好的模型文件 {model_save_path}")
    sys.exit(1)

model = UltimateRiskModel(input_dim=X.shape[1]).to(device)
# 注意：如果训练时只保存了 state_dict，使用 load_state_dict
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

with torch.no_grad():
    logits, _, _ = model(X_normalized, adj_list_full)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = Y.cpu().numpy()

# --- 7. 分行业评估 ---
def eval_sector(sector_mask, sector_name):
    # 取测试集与该行业掩码的交集索引
    test_indices_array = test_idx.numpy()
    sector_mask_array = sector_mask.numpy()
    target_idx = [i for i in test_indices_array if sector_mask_array[i]]
    
    if len(target_idx) == 0:
        print(f"⚠️ {sector_name}: 测试集中无该行业样本。")
        return
    
    y_s = y_true[target_idx]
    p_s = probs[target_idx]
    
    auc_score = roc_auc_score(y_s, p_s)
    precision, recall, _ = precision_recall_curve(y_s, p_s)
    pr_auc = auc(recall, precision)
    
    print(f"🎯 {sector_name:15} | 样本量: {len(target_idx):3} | AUC: {auc_score:.4f} | PR-AUC: {pr_auc:.4f}")

print("\n" + "="*60)
print("📊 ST-HyperGCL 行业差异化深度评估报告")
print("="*60)
eval_sector(masks['c38'], "C38 (电气机械)")
eval_sector(masks['c39'], "C39 (电子设备)")
print("="*60)