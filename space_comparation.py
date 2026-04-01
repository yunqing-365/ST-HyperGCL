import networkx as nx
import matplotlib.pyplot as plt
import random
from matplotlib.lines import Line2D

# ==========================================
# 0. 全局设置与颜色定义
# ==========================================
plt.rcParams['figure.dpi'] = 300
# 适当增加画布高度，给标题留出更多空间
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

color_normal = '#5C7E9E' # 宁静蓝
color_fraud = '#B25A57'  # 砖红色

# ==========================================
# 1. 左图：传统 GNN
# ==========================================
ax1 = axes[0]
G1 = nx.erdos_renyi_graph(n=80, p=0.12, seed=42)

random.seed(42)
colors1 = [color_normal if random.random() > 0.25 else color_fraud for _ in range(80)]
pos1 = nx.spring_layout(G1, seed=42, k=0.15)

nx.draw_networkx_nodes(G1, pos1, ax=ax1, node_color=colors1, node_size=120, edgecolors='white', linewidths=0.5, alpha=0.9)
nx.draw_networkx_edges(G1, pos1, ax=ax1, edge_color='gray', alpha=0.25)

# 关键修复 1：将 fontsize 调为 16，并使用 y=1.05 强制拉开与网络图的纵向距离
ax1.set_title("Traditional GNN:\nCatastrophic Over-smoothing", 
              fontsize=8, fontweight='bold', y=1.05, color='#333333')
ax1.axis('off')

# ==========================================
# 2. 右图：ST-HyperGCL 降噪后的潜空间
# ==========================================
ax2 = axes[1]
G2 = nx.Graph()

# 维持原有的健康企业“拓扑孤岛”
island1 = nx.erdos_renyi_graph(22, 0.35, seed=1)
island2 = nx.erdos_renyi_graph(28, 0.30, seed=2)
nx.relabel_nodes(island2, {i: i+22 for i in range(28)}, copy=False)
island3 = nx.erdos_renyi_graph(18, 0.40, seed=3)
nx.relabel_nodes(island3, {i: i+50 for i in range(18)}, copy=False)

G2.add_edges_from(island1.edges())
G2.add_edges_from(island2.edges())
G2.add_edges_from(island3.edges())

pos2 = {}
pos_i1 = nx.spring_layout(island1, center=(-2, 2), seed=1)
pos_i2 = nx.spring_layout(island2, center=(3, 0), seed=2)
pos_i3 = nx.spring_layout(island3, center=(-1, -3), seed=3)
pos2.update(pos_i1)
pos2.update(pos_i2)
pos2.update(pos_i3)

colors2 = [color_normal] * 68
fraud_nodes = range(68, 80)
G2.add_nodes_from(fraud_nodes)

# 维持原有的造假节点“寄生”边缘连接
G2.add_edge(68, 2);  G2.add_edge(68, 5)
G2.add_edge(69, 12); G2.add_edge(69, 19)
G2.add_edge(70, 22); G2.add_edge(70, 25)
G2.add_edge(71, 35); G2.add_edge(71, 40)
G2.add_edge(72, 44); G2.add_edge(73, 48)
G2.add_edge(74, 52); G2.add_edge(75, 58)
G2.add_edge(76, 60); G2.add_edge(76, 65)
G2.add_edge(77, 70); G2.add_edge(77, 22)
G2.add_edge(78, 68); G2.add_edge(79, 74)

colors2.extend([color_fraud] * 12)
pos2 = nx.spring_layout(G2, pos=pos2, fixed=range(68), seed=42, k=0.8)

nx.draw_networkx_nodes(G2, pos2, ax=ax2, node_color=colors2, node_size=120, edgecolors='white', linewidths=0.5, alpha=0.9)
nx.draw_networkx_edges(G2, pos2, ax=ax2, edge_color='gray', alpha=0.3)

# 关键修复 1 同上
ax2.set_title("Proposed Approach: Denoised Latent Space\n(Showing the Camouflage Paradox)", 
              fontsize=8, fontweight='bold', y=1.05, color='#333333')
ax2.axis('off')

# ==========================================
# 3. 添加图例与全局布局修复
# ==========================================
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color_normal, markersize=12, label='Normal (Healthy) Firm'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color_fraud, markersize=12, label='Fraud (Violating) Firm')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.02))

# 关键修复 2：放弃 tight_layout，使用 subplots_adjust 进行硬编码边距控制
# top=0.82 保证顶部留出 18% 的空白给标题；bottom=0.15 留出底端给图例
plt.subplots_adjust(top=0.82, bottom=0.15, wspace=0.1)

# ==========================================
# 4. 保存为高分辨率图片 (PDF 和 PNG)
# ==========================================
# 保存为 PDF (矢量图，顶会强烈推荐使用此格式排版，放大不失真)
plt.savefig("Figure_1_Latent_Space_Comparison.pdf", format='pdf', bbox_inches='tight', dpi=300)

# 保存为 PNG (位图，方便您预览和插入某些不支持 PDF 的文档)
plt.savefig("Figure_1_Latent_Space_Comparison.png", format='png', bbox_inches='tight', dpi=300)

# 显示图像
plt.show()