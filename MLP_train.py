#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLP 训练脚本
- 从 CSV 读取数据
- 丢弃非数值列
- LabelEncoder 编码 type
- 使用 SimpleImputer + StandardScaler 做预处理
- 训练 MLP 分类器
- 保存：
    - 模型参数：mlp_nellie_classifier.pt
    - 预处理器：mlp_preprocessor.joblib
    - LabelEncoder：mlp_label_encoder.joblib
    - 特征名：mlp_feature_names.npy
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import random

# ========== 路径 & 基本配置（按需修改） ==========
CSV_PATH = "/home/CWB/MTdata/MT_tiff/exp_all_train.csv"  # 训练数据路径
LABEL_COL = "type"                                       # 标签列名

MODEL_PATH = "./MLP/mlp_nellie_classifier.pt"
PREPROCESSOR_PATH = "./MLP/mlp_preprocessor.joblib"
LABEL_ENCODER_PATH = "./MLP/mlp_label_encoder.joblib"
FEATURE_NAMES_PATH = "./MLP/mlp_feature_names.npy"

BATCH_SIZE = 256
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VALID_SIZE = 0.2  # 验证集比例
RANDOM_SEED = 42


# ========== 为了可重复，设定随机种子 ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========== 定义 Dataset ==========
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        """
        X: np.ndarray, shape (N, D)
        y: np.ndarray, shape (N,)
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ========== 定义 MLP 结构 ==========
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
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # ========== 1. 读取数据 ==========
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到训练数据文件: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if LABEL_COL not in df.columns:
        raise ValueError(f"数据中找不到标签列: {LABEL_COL}")

    y_raw = df[LABEL_COL]
    X = df.drop(columns=[LABEL_COL])

    # ========== 2. 丢弃非数值列 ==========
    non_numeric_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_cols:
        print("发现非数值特征列，将丢弃:", non_numeric_cols)
        X = X.drop(columns=non_numeric_cols)

    feature_cols = X.columns.tolist()
    print(f"特征数: {len(feature_cols)}")

    X_values = X.values  # 仍可能有 NaN，稍后用 Imputer 处理

    # ========== 3. 标签编码 ==========
    le = LabelEncoder()
    y = le.fit_transform(y_raw.values)
    num_classes = len(le.classes_)
    print("类别:", list(le.classes_))

    # ========== 4. 划分训练 / 验证集 ==========
    X_train, X_val, y_train, y_val = train_test_split(
        X_values, y, test_size=VALID_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # ========== 5. 建立预处理 Pipeline：缺失值填充 + 标准化 ==========
    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # ========== 6. 建立 DataLoader ==========
    train_dataset = NumpyDataset(X_train_processed, y_train)
    val_dataset = NumpyDataset(X_val_processed, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ========== 7. 定义模型 / 损失 / 优化器 ==========
    input_dim = X_train_processed.shape[1]
    model = MLP(input_dim=input_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_val_acc = 0.0
    best_state_dict = None

    # ========== 8. 训练循环 ==========
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)              # (B, num_classes)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

            # 记录预测用于计算准确率
            preds = logits.argmax(dim=1)
            train_preds.append(preds.detach().cpu().numpy())
            train_targets.append(batch_y.detach().cpu().numpy())

        train_loss /= len(train_dataset)
        train_preds = np.concatenate(train_preds)
        train_targets = np.concatenate(train_targets)
        train_acc = accuracy_score(train_targets, train_preds)

        # ========== 9. 验证集 ==========
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_X)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_X.size(0)

                preds = logits.argmax(dim=1)
                val_preds.append(preds.detach().cpu().numpy())
                val_targets.append(batch_y.detach().cpu().numpy())

        val_loss /= len(val_dataset)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_acc = accuracy_score(val_targets, val_preds)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )

        # 记录最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    print(f"训练完成，最佳验证集准确率: {best_val_acc:.4f}")

    # ========== 10. 用最佳模型做一次详细报告 ==========
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    val_loader_full = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader_full:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    print("验证集分类报告：")
    print(
        classification_report(
            all_targets, all_preds, target_names=le.classes_
        )
    )

    # ========== 11. 保存模型 & 预处理器 & 编码器 & 特征名 ==========
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型参数已保存到: {MODEL_PATH}")

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"预处理器已保存到: {PREPROCESSOR_PATH}")

    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"LabelEncoder 已保存到: {LABEL_ENCODER_PATH}")

    np.save(FEATURE_NAMES_PATH, np.array(feature_cols))
    print(f"特征名已保存到: {FEATURE_NAMES_PATH}")


if __name__ == "__main__":
    main()

