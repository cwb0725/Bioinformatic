#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Nellie CSV 特征 + XGBoost 多分类示例脚本

使用方式：
    1. 修改下面的 CONFIG 区域中的 csv_path 和 label_col
    2. 在对应 conda 环境里运行：
       python train_xgb_nellie.py
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

# =========================
# 0. CONFIG：根据自己情况修改
# =========================
csv_path = "/home/CWB/MTdata/MT_tiff/exp_all_train_1.csv"  # Nellie 输出的 CSV 路径
label_col = "type"                    # 标签列名，例如: 'type', 'cell_line', etc.
test_size = 0.2                       # 测试集比例
random_state = 42                     # 随机种子，保证可复现

# =========================
# 1. 读取数据
# =========================
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

print(f"读取数据: {csv_path}")
df = pd.read_csv(csv_path)
print("原始数据形状:", df.shape)

if label_col not in df.columns:
    raise ValueError(f"标签列 '{label_col}' 不在数据中，请检查列名。")

# =========================
# 2. 基本预处理
# =========================

# 2.1 去掉全是 NaN 的列
df = df.dropna(axis=1, how="all")

# 2.2 去掉常数列（只有一个唯一值的列）
nunique = df.nunique()
constant_cols = nunique[nunique <= 1].index.tolist()
if constant_cols:
    print("去掉常数列:", constant_cols)
    df = df.drop(columns=constant_cols)

print("去掉全NaN列和常数列后数据形状:", df.shape)

# =========================
# 3. 提取特征 X 和标签 y
# =========================
y = df[label_col]
X = df.drop(columns=[label_col])

# 3.1 丢弃非数值特征列（后续想用可以专门做编码）
non_numeric_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
if non_numeric_cols:
    print("发现非数值特征列，将先丢弃:", non_numeric_cols)
    X = X.drop(columns=non_numeric_cols)

print("最终用于训练的特征数:", X.shape[1])

# =========================
# 4. 标签编码
# =========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 一维整型标签，例如 0,1,2,...

print("类别编码对照:")
for i, cls in enumerate(le.classes_):
    print(f"  {i} -> {cls}")

# =========================
# 5. 划分训练 / 测试集
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X.values,
    y_encoded,
    test_size=test_size,
    random_state=random_state,
    stratify=y_encoded,   # 保持各类比例
)

print("训练集形状:", X_train.shape, "测试集形状:", X_test.shape)

# =========================
# 6. 定义 XGBoost 模型
# =========================
num_classes = len(le.classes_)

model = XGBClassifier(
    n_estimators=300,        # 树的数量
    max_depth=6,            # 最大深度
    learning_rate=0.05,     # 学习率
    subsample=0.8,          # 行采样比例
    colsample_bytree=0.8,   # 列采样比例
    objective="multi:softprob",  # 多分类（输出各类概率）
    num_class=num_classes,
    eval_metric="mlogloss",
    tree_method="hist",     # CPU 版稳定可用；如支持 GPU 可改为 "gpu_hist"
    # predictor="gpu_predictor", # 如果你升级到 GPU 版 xgboost 再打开
)

print("\n开始训练 XGBoost 模型...")
model.fit(X_train, y_train)
print("训练完成。")

# =========================
# 7. 预测 & 评价
# =========================

# 7.1 概率输出（用于后续分析，可选）
y_proba = model.predict_proba(X_test)  # 形状 (N, num_classes)
print("预测概率矩阵形状:", y_proba.shape)

# 7.2 类别预测（用来算准确率和分类报告）
y_pred_raw = model.predict(X_test)     # 一般就是一维 (N,)

# 防呆：如果某些版本返回二维，这里统一压成一维
if y_pred_raw.ndim == 2:
    print("注意：predict 返回了二维，自动对 axis=1 做 argmax。")
    y_pred = np.argmax(y_pred_raw, axis=1)
else:
    y_pred = y_pred_raw

print("y_test shape:", y_test.shape)
print("y_pred shape:", y_pred.shape)

# 7.3 准确率
acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {acc:.4f}\n")

# 7.4 分类报告
print("分类报告:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# =========================
# 8. 特征重要性
# =========================
feature_importances = model.feature_importances_
feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": feature_importances
}).sort_values("importance", ascending=False)

print("\n最重要的前 30 个特征:")
print(feat_imp.head(30))

# 保存特征重要性
out_path = "xgb_feature_importance_1.csv"
feat_imp.to_csv(out_path, index=False)
print(f"\n特征重要性已保存到: {out_path}")

# =========================
# 9. 可选：保存模型
# =========================
model_out = "./xgb_model_1.json"
model.save_model(model_out)
print(f"模型已保存到: {model_out}")

