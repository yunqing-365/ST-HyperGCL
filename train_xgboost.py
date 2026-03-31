# 文件路径: train_xgboost.py

import torch
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
# 🌟 修改点 1：导入计算 PR-AUC 和 F1 的库
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, f1_score
import warnings
warnings.filterwarnings("ignore")

print("🔥 终极基线对决：加载纯财务静态特征...")

# 1. 加载相同的底层财务数据和标签
X = torch.load("data/processed/node_features_X.pt").cpu().numpy()
Y = torch.load("data/processed/risk_labels_Y.pt").cpu().numpy()

# 🌟 修改点 2：极其关键的标签对齐！保证 1=违规(少数类)，0=健康(多数类)，与 GNN 绝对一致
Y = 1 - Y  

# 2. 与超图模型保持绝对一致的特征标准化
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-5
X_normalized = (X - X_mean) / X_std

# 3. 应对违规样本稀缺的问题：自动计算正负样本比例，给违规样本加权重
num_negative = np.sum(Y == 0) # 健康企业数 (现在的 0)
num_positive = np.sum(Y == 1) # 违规企业数 (现在的 1)
scale_pos_weight_value = num_negative / num_positive

print(f"📊 数据分布: 总企业 {len(Y)} 家 | 健康(0) {num_negative} 家 | 违规(1) {num_positive} 家")

# 4. 祭出 XGBoost 终极树模型
clf = xgb.XGBClassifier(
    n_estimators=150,           # 树的数量
    max_depth=4,                # 控制树深，防止在极少违同样本上过拟合
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight_value, # 自动平衡类别权重
    subsample=0.8,              # 随机抽样比例，增强鲁棒性
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    n_jobs=-1                   # 调用所有 CPU 核心全速运行
)

# 5. 学术界最严谨的 5 折交叉验证 (5-Fold Cross Validation)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
acc_scores = []
# 🌟 修改点 3：新增 PR-AUC 和 F1 的记录列表
pr_auc_scores = []
f1_scores = []

print("\n🚀 开始 5 折交叉验证...")

for fold, (train_idx, test_idx) in enumerate(cv.split(X_normalized, Y)):
    X_train, X_test = X_normalized[train_idx], X_normalized[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]
    
    # 训练树模型
    clf.fit(X_train, y_train)
    
    # 预测概率 (用于计算 AUC 和 PR-AUC) 和 预测标签 (用于计算 Acc 和 F1)
    preds_prob = clf.predict_proba(X_test)[:, 1]
    preds_label = clf.predict(X_test)
    
    fold_auc = roc_auc_score(y_test, preds_prob)
    fold_acc = accuracy_score(y_test, preds_label)
    
    # 🌟 修改点 4：计算当前折的 PR-AUC 和 F1
    precision, recall, _ = precision_recall_curve(y_test, preds_prob)
    fold_pr_auc = auc(recall, precision)
    fold_f1 = f1_score(y_test, preds_label)
    
    auc_scores.append(fold_auc)
    acc_scores.append(fold_acc)
    pr_auc_scores.append(fold_pr_auc)
    f1_scores.append(fold_f1)
    
    print(f"Fold {fold+1} | AUC: {fold_auc:.3f} | PR-AUC: {fold_pr_auc:.3f} | F1: {fold_f1:.3f}")

# 6. 宣布最终成绩
mean_auc = np.mean(auc_scores)
mean_acc = np.mean(acc_scores)
mean_pr_auc = np.mean(pr_auc_scores)
mean_f1 = np.mean(f1_scores)

print("="*50)
print(f"🏆 XGBoost 最终平均成绩 (5-Fold CV):")
print(f"👑 AUC: {mean_auc:.3f} | 🏅 PR-AUC: {mean_pr_auc:.3f} | 🏅 F1-Score: {mean_f1:.3f}")
print("="*50)