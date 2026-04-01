import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 启动 XGBoost 基线测试 (绝对公平对比版)...")

# ==========================================
# 1. 加载数据 (完全对齐 V3 终极版)
# ==========================================
try:
    X = torch.load("data/processed/node_features_X.pt").cpu().numpy()
    Y = torch.load("data/processed/risk_labels_Y.pt").cpu().numpy().flatten()
except FileNotFoundError:
    print("❌ 找不到数据文件，请确认你在 ST-HyperGCL 根目录下运行！")
    exit()

num_companies = X.shape[0]

# ==========================================
# 2. 极其核心：严格复刻 V3 的划分逻辑！
# 必须保证 XGBoost 测试的公司，和 V3 测试的公司一模一样！
# ==========================================
torch.manual_seed(42)  # 连随机种子都必须和 V3 一模一样
indices = torch.randperm(num_companies).numpy()
split_point = int(0.8 * num_companies)
train_idx = indices[:split_point]
test_idx = indices[split_point:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = Y[train_idx], Y[test_idx]

# ==========================================
# 3. 应对极度不平衡数据的权重缩放
# ==========================================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ==========================================
# 4. 训练 XGBoost (传统风控之王)
# ==========================================
print(f"🌲 正在训练 XGBoost (仅使用纯静态财务特征)...")
clf = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc'
)
clf.fit(X_train, y_train)

# ==========================================
# 5. 预测与评估
# ==========================================
preds_prob = clf.predict_proba(X_test)[:, 1]
preds_bin = clf.predict(X_test)

auc_score = roc_auc_score(y_test, preds_prob)
precision, recall, _ = precision_recall_curve(y_test, preds_prob)
pr_auc = auc(recall, precision)
f1 = f1_score(y_test, preds_bin)

print("="*50)
print("📊 XGBoost 基线模型 (纯静态财务) OOT 盲测战报：")
print(f"   Test AUC    : {auc_score:.3f}")
print(f"   Test PR-AUC : {pr_auc:.3f}")
print(f"   Test F1     : {f1:.3f}")
print("="*50)