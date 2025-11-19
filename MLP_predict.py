#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLP 预测脚本
- 加载 mlp_train.py 训练好的模型 / 预处理器 / LabelEncoder / 特征名
- 对新的 CSV 数据做预测
- 输出类别编号、类别名称、最大预测概率
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib

# ========== 路径配置（按需修改） ==========
MODEL_PATH = "./MLP/mlp_nellie_classifier.pt"
PREPROCESSOR_PATH = "./MLP/mlp_preprocessor.joblib"
LABEL_ENCODER_PATH = "./MLP/mlp_label_encoder.joblib"
FEATURE_NAMES_PATH = "./MLP/mlp_feature_names.npy"

# 要预测的数据路径（修改这里）
CSV_PATH_NEW = "/home/CWB/MTdata/MT_tiff/exp_all_test_3.csv"
# 原始标签列名（如果新数据里也有，就会被丢掉）
LABEL_COL = "type"

OUT_CSV_PATH = "/home/CWB/MTdata/MT_tiff/exp_all_predict_with_mlp.csv"
BATCH_SIZE = 256


# ========== Dataset（只用来 batch 预测） ==========
class NumpyDataset(Dataset):
    def __init__(self, X):
        self.X = X.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


# ========== MLP 结构（必须和训练脚本完全一致） ==========
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # ========== 1. 加载预处理器 / LabelEncoder / 特征名 / 模型 ==========
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"找不到预处理器: {PREPROCESSOR_PATH}")
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"找不到 LabelEncoder: {LABEL_ENCODER_PATH}")
    if not os.path.exists(FEATURE_NAMES_PATH):
        raise FileNotFoundError(f"找不到特征名文件: {FEATURE_NAMES_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型参数: {MODEL_PATH}")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    feature_cols = np.load(FEATURE_NAMES_PATH, allow_pickle=True).tolist()

    num_classes = len(le.classes_)
    input_dim = len(feature_cols)

    model = MLP(input_dim=input_dim, num_classes=num_classes).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("类别:", list(le.classes_))
    print("特征数:", len(feature_cols))

    # ========== 2. 读取待预测数据 ==========
    if not os.path.exists(CSV_PATH_NEW):
        raise FileNotFoundError(f"找不到待预测数据: {CSV_PATH_NEW}")

    df_new = pd.read_csv(CSV_PATH_NEW)
    df_original = df_new.copy()  # 保留原始信息，方便一起输出

    # 如果新数据里也有标签列，先丢掉
    if LABEL_COL in df_new.columns:
        df_new = df_new.drop(columns=[LABEL_COL])

    # 丢掉训练没用到的额外列
    extra_cols = [c for c in df_new.columns if c not in feature_cols]
    if extra_cols:
        print("发现训练时未使用的列，将丢弃:", extra_cols)
        df_new = df_new.drop(columns=extra_cols)

    # 检查有没有缺失的特征列
    missing_cols = [c for c in feature_cols if c not in df_new.columns]
    if missing_cols:
        raise ValueError(
            f"待预测数据中缺少以下训练时使用的特征列: {missing_cols}"
        )

    # 按训练时顺序排列列
    df_new = df_new[feature_cols]

    X_new = df_new.values

    # ========== 3. 应用训练时的预处理（缺失值填充 + 标准化） ==========
    X_new_processed = preprocessor.transform(X_new)

    # ========== 4. 批量预测 ==========
    dataset_new = NumpyDataset(X_new_processed)
    loader_new = DataLoader(dataset_new, batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    all_preds_idx = []

    with torch.no_grad():
        for batch_X in loader_new:
            batch_X = batch_X.to(device)
            logits = model(batch_X)               # (B, num_classes)
            probs = torch.softmax(logits, dim=1)  # (B, num_classes)

            all_probs.append(probs.cpu().numpy())
            all_preds_idx.append(probs.argmax(dim=1).cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)         # (N, num_classes)
    all_preds_idx = np.concatenate(all_preds_idx, axis=0) # (N,)

    # 把类别索引映射回原始标签字符串
    all_preds_label = le.inverse_transform(all_preds_idx)
    max_prob = all_probs.max(axis=1)

    # ========== 5. 写回结果 ==========
    df_out = df_original.copy()
    df_out["mlp_pred_class_idx"] = all_preds_idx
    df_out["mlp_pred_class"] = all_preds_label
    df_out["mlp_pred_prob_max"] = max_prob

    # （可选）也可以输出每个类别的概率，比如：
    # for i, cls in enumerate(le.classes_):
    #     df_out[f"prob_{cls}"] = all_probs[:, i]

    df_out.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"预测完成，结果已保存到: {OUT_CSV_PATH}")


if __name__ == "__main__":
    main()

