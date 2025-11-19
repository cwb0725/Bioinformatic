#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TabTransformer full pipeline (v2.3.1, TRAIN)
- No internal numeric imputation/log/clip. Optional scaling only.
- Saves per-fold artifacts: preprocessor.json, cat_vocabs.json, (optional) scaler.pkl, model.pt, logs.
- Stable DataLoader defaults; AMP-safe stepping; robust AUC; CV summary.
- Supports --perm_imp / --perm_repeats for permutation importance.
"""

import os, json, random, argparse, pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------- Utils --------------------------

def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(bool(deterministic))
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -------------------------- Configs --------------------------

@dataclass
class PreprocConfig:
    target_col: str = "type"
    cat_cols: Optional[List[str]] = None
    num_cols: Optional[List[str]] = None
    scaler: str = "none"  # "none" | "standard" | "robust"
    rare_threshold: int = 10

class CatVocab:
    def __init__(self, rare_threshold: int = 10):
        self.rare_threshold = rare_threshold
        self.value2id: Dict[str, int] = {"<UNK>": 0}
        self.id2value: List[str] = ["<UNK>"]
        self._fitted = False

    def fit(self, series: pd.Series):
        vc = series.astype("object").value_counts(dropna=False)
        for val, cnt in vc.items():
            key = "<NA>" if pd.isna(val) else str(val)
            if cnt >= self.rare_threshold and key not in self.value2id:
                self.value2id[key] = len(self.value2id)
        if any(cnt < self.rare_threshold for cnt in vc.values) and "<RARE>" not in self.value2id:
            self.value2id["<RARE>"] = len(self.value2id)
        inv = sorted(self.value2id.items(), key=lambda x: x[1])
        self.id2value = [k for k, _ in inv]
        self._fitted = True

    def transform(self, series: pd.Series) -> np.ndarray:
        assert self._fitted
        out = []
        unk_id = self.value2id.get("<UNK>", 0)
        rare_id = self.value2id.get("<RARE>", unk_id)
        for val in series.astype("object").tolist():
            key = "<NA>" if pd.isna(val) else str(val)
            out.append(self.value2id.get(key, rare_id))
        return np.array(out, dtype=np.int64)

    def size(self) -> int:
        return len(self.value2id)

class TabularPreprocessor:
    def __init__(self, cfg: PreprocConfig):
        self.cfg = cfg
        self.cat_vocabs: Dict[str, CatVocab] = {}
        self.scaler = None
        self.fitted = False
        self._num_cols_real: List[str] = []

    @staticmethod
    def _auto_detect_columns(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
        cat_cols, num_cols = [], []
        N = len(df)
        threshold = max(50, int(0.05 * N))
        for c in df.columns:
            if c == target_col:
                continue
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                nun = s.nunique(dropna=True)
                if pd.api.types.is_integer_dtype(s) and nun <= threshold:
                    cat_cols.append(c)
                else:
                    num_cols.append(c)
            else:
                cat_cols.append(c)
        return cat_cols, num_cols

    def fit(self, df_train: pd.DataFrame):
        cfg = self.cfg
        cat_cols = cfg.cat_cols
        num_cols = cfg.num_cols
        if not cat_cols or not num_cols:
            auto_cat, auto_num = self._auto_detect_columns(df_train, cfg.target_col)
            if not cat_cols: cat_cols = auto_cat
            if not num_cols: num_cols = auto_num

        # drop obvious index-like columns from numeric
        num_cols = [c for c in num_cols if c.lower() not in {"unnamed: 0", "index", "idx"}]

        self.cat_cols, self.num_cols = cat_cols, num_cols
        self.cat_vocabs = {}
        for c in self.cat_cols:
            v = CatVocab(cfg.rare_threshold)
            v.fit(df_train[c])
            self.cat_vocabs[c] = v

        # Optional scaler fit (no imputation/log/clip)
        if cfg.scaler in ("standard", "robust") and len(self.num_cols) > 0:
            Xnum = df_train[self.num_cols].to_numpy(copy=True).astype(np.float32, copy=False)
            try:
                self.scaler = RobustScaler() if cfg.scaler == "robust" else StandardScaler()
                self.scaler.fit(Xnum)
            except Exception as e:
                print(f"[WARN] Scaler.fit failed (NaN/Inf?). Continue without scaling. Details: {e}")
                self.scaler = None
        else:
            self.scaler = None

        self._num_cols_real = list(self.num_cols)
        self.fitted = True

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        assert self.fitted, "Call fit() first"
        # Categorical
        cat_arrays = [self.cat_vocabs[c].transform(df[c]) for c in self.cat_cols] if self.cat_cols else []
        Xcat = np.vstack(cat_arrays).T if cat_arrays else np.zeros((len(df), 0), dtype=np.int64)

        # Numeric (pass-through; optional scaling only)
        if self._num_cols_real:
            Xnum = df[self._num_cols_real].to_numpy(copy=True).astype(np.float32, copy=False)
            if self.scaler is not None:
                try:
                    Xnum = self.scaler.transform(Xnum)
                except Exception as e:
                    print(f"[WARN] Scaler.transform failed; continue without scaling. Details: {e}")
        else:
            Xnum = np.zeros((len(df), 0), dtype=np.float32)

        return Xcat, Xnum

    def cardinalities(self) -> List[int]:
        return [self.cat_vocabs[c].size() for c in self.cat_cols]

    def numeric_dim(self) -> int:
        return len(self._num_cols_real)

# -------------------------- Dataset --------------------------

class TabDataset(Dataset):
    def __init__(self, Xcat, Xnum, y):
        if isinstance(Xcat, np.ndarray): Xcat = np.ascontiguousarray(Xcat).copy()
        if isinstance(Xnum, np.ndarray): Xnum = np.ascontiguousarray(Xnum).copy()
        if isinstance(y, np.ndarray): y = np.ascontiguousarray(y).copy()
        self.Xcat = torch.tensor(Xcat, dtype=torch.long) if Xcat.size else torch.zeros((len(y), 0), dtype=torch.long)
        self.Xnum = torch.tensor(Xnum, dtype=torch.float32) if Xnum.size else torch.zeros((len(y), 0), dtype=torch.float32)
        self.y    = torch.tensor(y,    dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.Xcat[i], self.Xnum[i], self.y[i]

# -------------------------- Model --------------------------

class TabTransformer(nn.Module):
    def __init__(self, cardinalities: List[int], num_dim: int, n_classes: int,
                 d_model: int = 128, n_heads: int = 4, depth: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.n_cat = len(cardinalities)
        self.n_num = num_dim
        self.d_model = d_model

        self.cat_embeddings = nn.ModuleList([nn.Embedding(card, d_model) for card in cardinalities])
        self.col_embeddings = nn.Embedding(self.n_cat, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True,
            dim_feedforward=4*d_model, dropout=p_drop,
            activation="gelu", norm_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.post_norm = nn.LayerNorm(d_model)

        hid = max(64, d_model)
        self.num_tower = (nn.Sequential(
            nn.LayerNorm(num_dim) if num_dim > 0 else nn.Identity(),
            nn.Linear(num_dim if num_dim > 0 else 1, hid),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hid, d_model)
        ) if num_dim > 0 else None)

        out_in = d_model * (2 if num_dim > 0 else 1)
        self.classifier = nn.Sequential(
            nn.Linear(out_in, d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, xcat: torch.Tensor, xnum: torch.Tensor):
        B = xcat.size(0)
        if self.n_cat > 0:
            toks = []
            for j, emb in enumerate(self.cat_embeddings):
                t = emb(xcat[:, j]) + self.col_embeddings(torch.full((B,), j, device=xcat.device))
                toks.append(t.unsqueeze(1))
            cat_tokens = torch.cat(toks, dim=1)
            seq = torch.cat([self.cls.expand(B, 1, -1), cat_tokens], dim=1)
        else:
            seq = self.cls.expand(B, 1, -1)
        h = self.encoder(seq)
        h_cls = self.post_norm(h[:, 0, :])
        if self.num_tower is not None and xnum is not None and xnum.shape[1] > 0:
            h_num = self.num_tower(xnum)
            h_all = torch.cat([h_cls, h_num], dim=1)
        else:
            h_all = h_cls
        return self.classifier(h_all)

# -------------------------- Training --------------------------

@dataclass
class TrainConfig:
    n_folds: int = 5
    epochs: int = 80
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-5
    dropout: float = 0.05
    d_model: int = 256
    depth: int = 4
    n_heads: int = 8
    label_smoothing: float = 0.0
    patience: int = 20
    amp: bool = False
    amp_dtype: str = "fp16"  # "fp16" or "bf16"
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 42
    deterministic: bool = False
    do_permutation_importance: bool = False
    perm_repeats: int = 1

class EarlyStopper:
    def __init__(self, patience: int = 15, mode: str = "min"):
        self.best_score = None
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_state = None
    def step(self, score, model):
        improved = (self.best_score is None) or \
                   ((self.mode == "min" and score < self.best_score) or
                    (self.mode == "max" and score > self.best_score))
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return True
        else:
            self.counter += 1
            return False
    def should_stop(self):
        return self.counter >= self.patience

def build_onecycle(optimizer, max_lr, epochs, steps_per_epoch, pct_start=0.1, div_factor=25.0, final_div_factor=1e4):
    try:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
            pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor, anneal_strategy='cos'
        )
    except TypeError:
        total_steps = max(1, epochs * steps_per_epoch)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps,
            pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor, anneal_strategy='cos'
        )
    return sched

@torch.no_grad()
def evaluate_loss(model, loader, device, label_smoothing: float = 0.0, amp_enabled: bool = False, amp_dtype="fp16"):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    model.eval(); tot, n = 0.0, 0
    use_cuda = (device == "cuda")
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    for xcat, xnum, yb in loader:
        xcat, xnum, yb = xcat.to(device), xnum.to(device), yb.to(device)
        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=(amp_enabled and use_cuda))
        with ctx:
            logits = model(xcat, xnum)
            loss = criterion(logits, yb)
        tot += loss.item() * yb.size(0); n += yb.size(0)
    return tot / max(1, n)

def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=2 if int(num_workers) > 0 else None,
        timeout=120 if int(num_workers) > 0 else 0
    )

def run_fold(fold_id: int, df: pd.DataFrame, pre_cfg: PreprocConfig, tr_cfg: TrainConfig,
             outdir: str, device: str, class_names: List[str]):
    ensure_dir(outdir)
    n_classes = len(class_names)

    # split
    y_codes = df[pre_cfg.target_col].astype("category").cat.codes.values
    skf = StratifiedKFold(n_splits=tr_cfg.n_folds, shuffle=True, random_state=tr_cfg.seed)
    train_idx, valid_idx = list(skf.split(df, y_codes))[fold_id]
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)

    # prep
    prep = TabularPreprocessor(pre_cfg); prep.fit(df_train)
    Xcat_tr, Xnum_tr = prep.transform(df_train)
    Xcat_va, Xnum_va = prep.transform(df_valid)

    y_tr = df_train[pre_cfg.target_col].astype("category").cat.codes.to_numpy(copy=True)
    y_va = df_valid[pre_cfg.target_col].astype("category").cat.codes.to_numpy(copy=True)

    train_ds = TabDataset(Xcat_tr, Xnum_tr, y_tr)
    valid_ds = TabDataset(Xcat_va, Xnum_va, y_va)

    train_loader = make_loader(train_ds, tr_cfg.batch_size, True, tr_cfg.num_workers, tr_cfg.pin_memory)
    valid_loader = make_loader(valid_ds, tr_cfg.batch_size, False, tr_cfg.num_workers, tr_cfg.pin_memory)

    # model
    model = TabTransformer(
        prep.cardinalities(), prep.numeric_dim(), n_classes,
        d_model=tr_cfg.d_model, n_heads=tr_cfg.n_heads, depth=tr_cfg.depth, p_drop=tr_cfg.dropout
    ).to(device)

    # optim & sched
    criterion = nn.CrossEntropyLoss(label_smoothing=tr_cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr_cfg.lr, weight_decay=tr_cfg.weight_decay)
    scheduler = build_onecycle(optimizer, max_lr=tr_cfg.lr, epochs=tr_cfg.epochs, steps_per_epoch=max(1, len(train_loader)))

    # AMP
    use_cuda = (device == "cuda")
    amp_enabled = bool(tr_cfg.amp) and use_cuda
    amp_dtype = torch.float16 if tr_cfg.amp_dtype.lower() == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    early = EarlyStopper(patience=tr_cfg.patience, mode="min")
    log_rows = []

    for epoch in range(1, tr_cfg.epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0

        for xcat, xnum, yb in train_loader:
            xcat, xnum, yb = xcat.to(device), xnum.to(device), yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
            with ctx:
                logits = model(xcat, xnum)
                loss = criterion(logits, yb)

            prev_scale = scaler.get_scale() if amp_enabled else None
            scaler.scale(loss).backward()
            try:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            except Exception:
                pass
            scaler.step(optimizer)
            scaler.update()

            did_step = (not amp_enabled) or (scaler.get_scale() >= prev_scale)
            if did_step:
                scheduler.step()

            tr_loss += loss.item() * yb.size(0); tr_n += yb.size(0)

        # valid
        model.eval()
        va_loss, va_n = 0.0, 0
        preds, probs, labels = [], [], []
        with torch.no_grad():
            for xcat, xnum, yb in valid_loader:
                xcat, xnum, yb = xcat.to(device), xnum.to(device), yb.to(device)
                ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
                with ctx:
                    logits = model(xcat, xnum)
                    loss = criterion(logits, yb)
                va_loss += loss.item() * yb.size(0); va_n += yb.size(0)
                p = torch.softmax(logits, dim=1)
                probs.append(p.detach().cpu().numpy())
                preds.append(p.argmax(1).detach().cpu().numpy())
                labels.append(yb.detach().cpu().numpy())

        tr_loss /= max(1, tr_n); va_loss /= max(1, va_n)
        probs = np.concatenate(probs, axis=0) if probs else np.zeros((len(valid_ds), n_classes))
        preds = np.concatenate(preds, axis=0) if preds else np.zeros((len(valid_ds),), dtype=int)
        labels = np.concatenate(labels, axis=0) if labels else np.zeros((len(valid_ds),), dtype=int)

        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")

        macro_auc = float("nan")
        try:
            if n_classes == 2:
                if len(np.unique(labels)) == 2:
                    macro_auc = roc_auc_score(labels, probs[:, 1])
            else:
                macro_auc = roc_auc_score(labels, probs, multi_class="ovr")
        except Exception:
            macro_auc = float("nan")

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "valid_loss": va_loss,
            "acc": acc,
            "macro_f1": macro_f1,
            "macro_auc": macro_auc,
        }
        log_rows.append(row)
        print(f"[Fold {fold_id}] Epoch {epoch:02d} | train {tr_loss:.4f} | valid {va_loss:.4f} | acc {acc:.3f} | F1 {macro_f1:.3f} | AUC {macro_auc:.3f}")

        early.step(va_loss, model)
        if early.should_stop():
            print(f"[Fold {fold_id}] Early stopping at epoch {epoch}")
            break

    if early.best_state is not None:
        model.load_state_dict(early.best_state)

    fold_dir = os.path.join(outdir, f"fold_{fold_id}")
    ensure_dir(fold_dir)
    # Save logs & model
    pd.DataFrame(log_rows).to_csv(os.path.join(fold_dir, "train_log.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))

    # Save preprocessor summary
    pre_sum = {
        "cat_cols": getattr(prep, "cat_cols", []),
        "num_cols": getattr(prep, "num_cols", []),
        "cardinalities": prep.cardinalities(),
        "numeric_dim": prep.numeric_dim(),
        "scaler": pre_cfg.scaler
    }
    save_json(pre_sum, os.path.join(fold_dir, "preprocessor.json"))

    # Save cat vocabs (value->id) for exact mapping at inference
    cat_vocabs = {col: vocab.value2id for col, vocab in prep.cat_vocabs.items()}
    save_json(cat_vocabs, os.path.join(fold_dir, "cat_vocabs.json"))

    # Save scaler (if any)
    if pre_cfg.scaler in ("standard", "robust") and prep.scaler is not None:
        with open(os.path.join(fold_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(prep.scaler, f)

    # Save final metrics
    save_json(log_rows[-1], os.path.join(fold_dir, "final_metrics.json"))

    # Permutation importance (optional)
    if tr_cfg.do_permutation_importance:
        print(f"[Fold {fold_id}] Computing permutation importance...")
        baseline = evaluate_loss(model, valid_loader, device, label_smoothing=tr_cfg.label_smoothing,
                                 amp_enabled=bool(tr_cfg.amp), amp_dtype=("fp16" if tr_cfg.amp_dtype.lower()=="fp16" else "bf16"))
        rows = []

        # categorical columns
        if getattr(prep, "cat_cols", []):
            Xc_base = Xcat_va.copy()
            for j, col in enumerate(prep.cat_cols):
                losses = []
                for r in range(tr_cfg.perm_repeats):
                    Xc = Xc_base.copy()
                    perm = np.random.permutation(Xc.shape[0])
                    Xc[:, j] = Xc[perm, j]
                    dl = make_loader(TabDataset(Xc, Xnum_va, y_va), tr_cfg.batch_size, False, tr_cfg.num_workers, tr_cfg.pin_memory)
                    losses.append(evaluate_loss(model, dl, device, label_smoothing=tr_cfg.label_smoothing,
                                                amp_enabled=bool(tr_cfg.amp), amp_dtype=("fp16" if tr_cfg.amp_dtype.lower()=="fp16" else "bf16")))
                rows.append({"column": col, "type": "categorical", "delta_valid_loss": float(np.mean(losses) - baseline)})
                if (j+1) % 10 == 0:
                    print(f"... categorical {j+1}/{len(prep.cat_cols)} done")

        # numeric columns
        if getattr(prep, "_num_cols_real", []):
            Xn_base = Xnum_va.copy()
            for j, col in enumerate(prep._num_cols_real):
                losses = []
                for r in range(tr_cfg.perm_repeats):
                    Xn = Xn_base.copy()
                    perm = np.random.permutation(Xn.shape[0])
                    Xn[:, j] = Xn[perm, j]
                    dl = make_loader(TabDataset(Xcat_va, Xn, y_va), tr_cfg.batch_size, False, tr_cfg.num_workers, tr_cfg.pin_memory)
                    losses.append(evaluate_loss(model, dl, device, label_smoothing=tr_cfg.label_smoothing,
                                                amp_enabled=bool(tr_cfg.amp), amp_dtype=("fp16" if tr_cfg.amp_dtype.lower()=="fp16" else "bf16")))
                rows.append({"column": col, "type": "numeric", "delta_valid_loss": float(np.mean(losses) - baseline)})
                if (j+1) % 20 == 0:
                    print(f"... numeric {j+1}/{len(prep._num_cols_real)} done")

        pd.DataFrame(rows).sort_values("delta_valid_loss", ascending=False)\
          .to_csv(os.path.join(fold_dir, "permutation_importance.csv"), index=False)

    return log_rows[-1]

# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--target", type=str, default="type")
    ap.add_argument("--cat_cols", type=str, default="")
    ap.add_argument("--num_cols", type=str, default="")
    ap.add_argument("--rare_threshold", type=int, default=10)
    ap.add_argument("--scaler", type=str, default="none", choices=["none", "standard", "robust"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16","bf16"])
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--perm_imp", action="store_true")
    ap.add_argument("--perm_repeats", type=int, default=1)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    # Save global configs at root
    pre_cfg = PreprocConfig(target_col=args.target,
                            cat_cols=[c for c in args.cat_cols.split(",") if c] if args.cat_cols else None,
                            num_cols=[c for c in args.num_cols.split(",") if c] if args.num_cols else None,
                            scaler=args.scaler, rare_threshold=args.rare_threshold)
    save_json(asdict(pre_cfg), os.path.join(args.outdir, "preprocess_config.json"))

    tr_cfg = TrainConfig(n_folds=args.folds, epochs=args.epochs, batch_size=args.batch_size,
                         lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
                         d_model=args.d_model, depth=args.depth, n_heads=args.heads,
                         label_smoothing=args.label_smoothing, patience=args.patience,
                         amp=(not args.no_amp), amp_dtype=args.amp_dtype,
                         num_workers=args.num_workers, pin_memory=args.pin_memory,
                         seed=args.seed, deterministic=args.deterministic,
                         do_permutation_importance=args.perm_imp, perm_repeats=args.perm_repeats)
    save_json(asdict(tr_cfg), os.path.join(args.outdir, "train_config.json"))

    set_seed(tr_cfg.seed, deterministic=tr_cfg.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.data)
    assert args.target in df.columns, f"Target column '{args.target}' not in data"
    df = df.dropna(subset=[args.target]).reset_index(drop=True)
    df[args.target] = df[args.target].astype("category")
    class_names = list(df[args.target].cat.categories)
    save_json({"class_names": class_names}, os.path.join(args.outdir, "class_names.json"))

    fold_metrics = []
    for fold in range(tr_cfg.n_folds):
        print(f"\n========== Fold {fold}/{tr_cfg.n_folds} ==========")
        m = run_fold(fold, df, pre_cfg, tr_cfg, args.outdir, device, class_names)
        m["fold"] = fold
        fold_metrics.append(m)

    agg = pd.DataFrame(fold_metrics)
    agg.to_csv(os.path.join(args.outdir, "cv_metrics.csv"), index=False)

    # Safe CV summary
    summary = {}
    for k in ["valid_loss", "acc", "macro_f1", "macro_auc"]:
        if k in agg.columns:
            vals = agg[k].dropna().values
            if len(vals) > 0:
                summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    save_json(summary, os.path.join(args.outdir, "cv_summary.json"))
    print("\nCV Summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
