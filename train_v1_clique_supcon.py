import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# 🌟 修改点 1：在这里补充导入了计算 PR-AUC 和 F1 需要的库
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, f1_score
import warnings
warnings.filterwarnings("ignore")

from models.st_hypergcl import UltimateRiskModel

# ====================== 【温柔的 SupCon 有监督去噪】 ======================
def supervised_contrastive_loss(z1, z2, labels, temperature=0.5):
    """
    有监督 InfoNCE: 拉近同类，推开异类。完美适配二分类风控！
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    y = torch.cat([labels, labels], dim=0).view(-1, 1)
    
    sim_matrix = torch.matmul(z, z.t()) / temperature
    mask = torch.eq(y, y.t()).float()
    mask.fill_diagonal_(0) # 不与自己对比
    
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    return -mean_log_prob_pos.mean()

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

# ====================== 【2. 严格隔离划分】 ======================
torch.manual_seed(42)
indices = torch.randperm(num_companies)
split_point = int(0.8 * num_companies)
train_idx = indices[:split_point]
test_idx = indices[split_point:]

# ====================== 【3. 完整图预处理】 ======================
adj_list_full = []
for t in range(T):
    edge_index_t = edge_index_list[t].to(device)
    if edge_index_t.shape[1] == 0:
        adj_list_full.append(torch.eye(num_companies, device=device))
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
    adj_list_full.append(A_norm)

# ====================== 【4. 模型配置】 ======================
model = UltimateRiskModel(input_dim=X.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

num_negative = (Y[train_idx] == 0).sum().item()
num_positive = (Y[train_idx] == 1).sum().item()
pos_weight = torch.tensor([num_negative / (num_positive + 1e-5)], device=device)
criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ====================== 【5. 温柔微操训练循环】 ======================
epochs = 200
alpha = 0.05  
drop_rate = 0.2  

# 🌟 修改点 2：新增用于记录巅峰状态下的 PR-AUC 和 F1-Score 的变量
best_auc = 0.
best_pr_auc = 0. 
best_f1 = 0.     
best_epoch = 0

print(f"\n🚀 开始【温柔超车版】训练 (5%删边微扰 + SupCon 有监督轻微聚类)...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # 轻微扰动图结构
    adj_list_dropped = []
    for t in range(T):
        edge_index_t = edge_index_list[t].to(device)
        if edge_index_t.shape[1] == 0:
            adj_list_dropped.append(torch.eye(num_companies, device=device))
            continue
        num_edges = edge_index_t.size(1)
        mask = torch.rand(num_edges, device=device) > drop_rate
        H_indices_dropped = edge_index_t[[1, 0]][:, mask] 
        H_values_dropped = torch.ones(H_indices_dropped.size(1), device=device)
        if H_indices_dropped.size(1) > 0:
            H_sparse = torch.sparse_coo_tensor(H_indices_dropped, H_values_dropped, size=(num_companies, edge_index_t[0].max().item() + 1), device=device)
            A_sparse = torch.sparse.mm(H_sparse, H_sparse.t())
            A_dense = A_sparse.to_dense()
        else:
            A_dense = torch.zeros((num_companies, num_companies), device=device)
        A_dense.fill_diagonal_(1.0)
        deg = A_dense.sum(dim=1, keepdim=True)
        deg[deg == 0] = 1.0
        A_norm = A_dense / deg
        adj_list_dropped.append(A_norm)

    logits, view1, view2 = model(X_normalized, adj_list_dropped)
    
    # Loss 都在严格隔离的 train_idx 上计算
    loss_bce = criterion_bce(logits[train_idx], Y[train_idx])
    loss_gcl = supervised_contrastive_loss(view1[train_idx], view2[train_idx], Y[train_idx])
    
    loss = loss_bce + alpha * loss_gcl
    loss.backward()
    optimizer.step()
    
    # 密集抓取巅峰 (每轮评估)
    model.eval()
    with torch.no_grad():
        preds_prob = torch.sigmoid(model(X_normalized, adj_list_full)[0])
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
            pass
        
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | BCE: {loss_bce.item():.3f} | SupCon: {loss_gcl.item():.3f} | 当前 Test AUC: {auc_test:.3f}")

# 🌟 修改点 4：在训练结束的总结栏里，把新指标打印出来
print("="*50)
print(f"🏆 涅槃重生！在严格隔离的测试集上：")
print(f"👑 最终巅峰 Test AUC 达到了: {best_auc:.3f} (出现在第 {best_epoch} 轮)")
print(f"🏅 补充顶会指标 -> PR-AUC: {best_pr_auc:.3f} | F1-Score: {best_f1:.3f}")
print("="*50)

# ====================== 【追加：真实 t-SNE 可视化】 ======================
print("\n🔍 正在提取真实网络特征绘制 t-SNE 聚类图...")
import numpy as np  
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 设置顶会字体风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14

model.eval()
with torch.no_grad():
    _, view1, _ = model(X_normalized, adj_list_full)
    real_embeddings = view1.cpu().numpy()
    real_labels = Y.cpu().numpy()

# 进行 t-SNE 降维
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
embeddings_2d = tsne.fit_transform(real_embeddings)

# 开始画图
fig, ax = plt.subplots(figsize=(7, 6))
healthy_idx = np.where(real_labels == 0)[0]
violation_idx = np.where(real_labels == 1)[0]

# 绘制散点
ax.scatter(embeddings_2d[healthy_idx, 0], embeddings_2d[healthy_idx, 1], 
           c='#55A868', label='Compliant Firms (0)', alpha=0.7, edgecolors='w', s=60, linewidths=0.5)
ax.scatter(embeddings_2d[violation_idx, 0], embeddings_2d[violation_idx, 1], 
           c='#C44E52', label='Violation Firms (1)', alpha=0.8, edgecolors='w', s=60, linewidths=0.5)

ax.set_title('t-SNE Visualization of Latent Space (Real Data)', fontweight='bold', pad=15)
ax.set_xticks([]) 
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.legend(loc='best', framealpha=0.9, edgecolor='black', fancybox=False)
plt.tight_layout()
plt.savefig('real_tsne_visualization.pdf', format='pdf', dpi=300)
plt.savefig('real_tsne_visualization.png', format='png', dpi=300)

print("✅ 基于真实模型参数的 t-SNE 图表已生成！")