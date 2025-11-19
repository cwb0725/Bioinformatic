#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fast_explain_hybrid_v231_exportall.py
Export ALL feature scores (per sample × per feature) for TabTransformer v2.3.1

- 计算与“稳健加速版 hybrid”一致：
  * categorical: Δprob（整列替换为 <UNK>，仅看预测类）
  * numeric: Captum IG（只对数值塔；baseline=0/均值）
  * 对两类分别做列内 |value| 均值缩放（保符号），再合并为最终 score
- 导出：
  * scores_wide.csv  （含 id 列、pred_class、base_prob 与所有特征分数）
  * scores_long.csv  （长表：row_index, feature, score，可选并发分析）
  * （可选）raw_cat_deltas.npy、raw_num_ig.npy、scores.npy

Usage:
  pip install -U captum
  python fast_explain_hybrid_v231_exportall.py \
    --model_outdir ./MTtiff/exp_all_out_v231 \
    --input_csv    ./MT_tiff/new_samples.csv \
    --output_dir   ./MT_tiff/explain_all \
    --id_cols id,t \
    --fold fold_0 \
    --batch_size 256 \
    --micro_batch_size 64 \
    --ig_steps 16 \
    --no_amp \
    --save_raw
"""

import os, json, argparse, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- Model ----------------
class TabTransformer(nn.Module):
    def __init__(self, cardinalities, num_dim, n_classes, d_model=256, n_heads=8, depth=4, p_drop=0.05):
        super().__init__()
        self.n_cat=len(cardinalities); self.n_num=num_dim; self.d_model=d_model
        self.cat_embeddings=nn.ModuleList([nn.Embedding(card, d_model) for card in cardinalities])
        self.col_embeddings=nn.Embedding(self.n_cat, d_model)
        self.cls=nn.Parameter(torch.zeros(1,1,d_model))
        enc_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True,
                    dim_feedforward=4*d_model, dropout=p_drop, activation="gelu", norm_first=False)
        self.encoder=nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.post_norm=nn.LayerNorm(d_model)
        hid=max(64,d_model)
        self.num_tower=(nn.Sequential(
            nn.LayerNorm(num_dim) if num_dim>0 else nn.Identity(),
            nn.Linear(num_dim if num_dim>0 else 1, hid), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hid, d_model)
        ) if num_dim>0 else None)
        out_in=d_model*(2 if num_dim>0 else 1)
        self.classifier=nn.Sequential(nn.Linear(out_in, d_model), nn.GELU(), nn.Dropout(p_drop), nn.Linear(d_model, n_classes))
    def forward(self, xcat, xnum):
        B=xcat.size(0)
        if self.n_cat>0:
            toks=[]
            for j,emb in enumerate(self.cat_embeddings):
                t=emb(xcat[:,j]) + self.col_embeddings(torch.full((B,), j, device=xcat.device))
                toks.append(t.unsqueeze(1))
            cat_tokens=torch.cat(toks, dim=1)
            seq=torch.cat([self.cls.expand(B,1,-1), cat_tokens], dim=1)
        else:
            seq=self.cls.expand(B,1,-1)
        h=self.encoder(seq); h_cls=self.post_norm(h[:,0,:])
        if self.num_tower is not None and xnum is not None and xnum.shape[1]>0:
            h_num=self.num_tower(xnum); h_all=torch.cat([h_cls,h_num], dim=1)
        else:
            h_all=h_cls
        return self.classifier(h_all)

# ---------------- Helpers ----------------
def load_json(p):
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def map_categoricals(df_in: pd.DataFrame, cat_cols, vocabs: dict) -> np.ndarray:
    X_list=[]
    for col in cat_cols:
        vocab=vocabs.get(col, {"<UNK>":0})
        unk_id=vocab.get("<UNK>", 0); rare_id=vocab.get("<RARE>", unk_id)
        series=df_in[col] if col in df_in.columns else pd.Series([pd.NA]*len(df_in))
        vals=[]
        for v in series.astype("object").tolist():
            key="<NA>" if pd.isna(v) else str(v)
            vals.append(vocab.get(key, rare_id))
        X_list.append(np.array(vals, dtype=np.int64))
    return np.vstack(X_list).T if X_list else np.zeros((len(df_in),0), dtype=np.int64)

def maybe_scale_numeric(Xnum: np.ndarray, scaler_path: str):
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler=pickle.load(f)
            return scaler.transform(Xnum.astype(np.float32, copy=False))
        except Exception as e:
            print(f"[WARN] Failed to apply scaler from {scaler_path}: {e}")
    return Xnum.astype(np.float32, copy=False)

def zscore_per_col(a: np.ndarray) -> np.ndarray:
    if a.size == 0: return a
    m = np.mean(np.abs(a), axis=0, keepdims=True) + 1e-12
    return a / m  # 保留符号，只缩放幅度，便于跨列可比

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_outdir", type=str, required=True)
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--id_cols", type=str, default="")
    ap.add_argument("--fold", type=str, default="fold_0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--micro_batch_size", type=int, default=64, help="internal batch for IG")
    ap.add_argument("--ig_steps", type=int, default=16)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16","bf16"])
    ap.add_argument("--save_raw", action="store_true", help="save raw_cat_deltas.npy, raw_num_ig.npy, scores.npy")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.device=="cuda": device="cuda"
    elif args.device=="cpu": device="cpu"
    else: device="cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled=(not args.no_amp) and (device=="cuda")
    amp_dtype=torch.float16 if args.amp_dtype=="fp16" else torch.bfloat16

    # Artifacts
    classes = load_json(os.path.join(args.model_outdir, "class_names.json"))["class_names"]
    fold_dir = os.path.join(args.model_outdir, args.fold)
    if not os.path.isdir(fold_dir): raise RuntimeError(f"Fold dir not found: "+fold_dir)
    pre = load_json(os.path.join(fold_dir, "preprocessor.json"))
    voc = load_json(os.path.join(fold_dir, "cat_vocabs.json"))
    cat_cols = pre.get("cat_cols", []); num_cols = pre.get("num_cols", [])
    cardinalities = pre.get("cardinalities", []); num_dim = pre.get("numeric_dim", len(num_cols))

    # Hyperparams
    d_model=256; depth=4; heads=8; pdrop=0.05
    try:
        tr_cfg = load_json(os.path.join(args.model_outdir, "train_config.json"))
        d_model = tr_cfg.get("d_model", d_model); depth = tr_cfg.get("depth", depth)
        heads   = tr_cfg.get("n_heads", heads); pdrop  = tr_cfg.get("dropout", pdrop)
    except Exception: pass

    mdl = TabTransformer(cardinalities, num_dim, len(classes),
                         d_model=d_model, n_heads=heads, depth=depth, p_drop=pdrop).to(device).eval()
    state = torch.load(os.path.join(fold_dir, "model.pt"), map_location=device, weights_only=True)
    mdl.load_state_dict(state)

    # Data
    df = pd.read_csv(args.input_csv)
    for c in cat_cols:
        if c not in df.columns: df[c]=pd.NA
    for c in num_cols:
        if c not in df.columns: df[c]=np.nan
    Xcat = map_categoricals(df, cat_cols, voc)
    Xnum_raw = df[num_cols].to_numpy(copy=True) if num_cols else np.zeros((len(df),0), dtype=np.float32)
    Xnum = maybe_scale_numeric(Xnum_raw, os.path.join(fold_dir, "scaler.pkl"))

    Xcat_t = torch.tensor(Xcat, dtype=torch.long, device=device)
    Xnum_t = torch.tensor(Xnum, dtype=torch.float32, device=device)

    # Base predictions
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(df), args.batch_size):
            xb_cat = Xcat_t[i:i+args.batch_size]; xb_num = Xnum_t[i:i+args.batch_size]
            ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
            with ctx:
                logits = mdl(xb_cat, xb_num)
                probs = torch.softmax(logits, dim=1)
            probs_all.append(probs)
    probs = torch.cat(probs_all, dim=0).detach().cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    pred_cls = [classes[i] for i in pred_idx]

    # A) Categorical：Δprob
    cat_deltas = np.zeros((len(df), len(cat_cols)), dtype=np.float32) if len(cat_cols)>0 else np.zeros((len(df), 0), dtype=np.float32)
    if len(cat_cols) > 0:
        unk_ids = [voc.get(col, {"<UNK>":0}).get("<UNK>", 0) for col in cat_cols]
        for j in range(len(cat_cols)):
            Xcat_mut = Xcat_t.clone()
            Xcat_mut[:, j] = int(unk_ids[j])
            deltas = []
            with torch.no_grad():
                for i in range(0, len(df), args.batch_size):
                    xb_cat = Xcat_mut[i:i+args.batch_size]; xb_num = Xnum_t[i:i+args.batch_size]
                    ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
                    with ctx:
                        p = torch.softmax(mdl(xb_cat, xb_num), dim=1).detach().cpu().numpy()
                    base = probs[i:i+args.batch_size, :]
                    sel = pred_idx[i:i+args.batch_size]
                    deltas.append(base[np.arange(len(sel)), sel] - p[np.arange(len(sel)), sel])
            cat_deltas[:, j] = np.concatenate(deltas, axis=0)

    # B) Numeric：IG（带降级/CPU回退）
    if len(num_cols) > 0:
        try:
            from captum.attr import IntegratedGradients
        except Exception as e:
            raise RuntimeError("Please install Captum: pip install -U captum") from e

        class NumOnlyWrapper(nn.Module):
            def __init__(self, mdl: nn.Module, xcat_fixed: torch.Tensor, amp_enabled: bool, amp_dtype: torch.dtype):
                super().__init__(); self.mdl=mdl; self.xcat=xcat_fixed
                self.amp_enabled=amp_enabled; self.amp_dtype=amp_dtype
            def forward(self, xnum):
                ctx = torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled)
                with ctx:
                    return self.mdl(self.xcat, xnum)

        num_attrib = np.zeros((len(df), len(num_cols)), dtype=np.float32)
        for i in range(0, len(df), args.batch_size):
            xb_cat = Xcat_t[i:i+args.batch_size]
            xb_num = Xnum_t[i:i+args.batch_size]
            targets = torch.tensor(pred_idx[i:i+args.batch_size], dtype=torch.long, device=device)
            try:
                xb_num1 = xb_num.clone().detach().requires_grad_(True)
                wrap = NumOnlyWrapper(mdl, xb_cat, amp_enabled, amp_dtype).to(device).eval()
                ig = IntegratedGradients(wrap)
                attr = ig.attribute(xb_num1, baselines=torch.zeros_like(xb_num1),
                                    target=targets, n_steps=int(args.ig_steps), internal_batch_size=int(args.micro_batch_size))
                num_attrib[i:i+args.batch_size] = attr.detach().cpu().numpy()
                continue
            except RuntimeError:
                # 降 internal_batch_size → CPU 兜底
                try:
                    xb_num2 = xb_num.clone().detach().requires_grad_(True)
                    wrap = NumOnlyWrapper(mdl, xb_cat, amp_enabled, amp_dtype).to(device).eval()
                    ig = IntegratedGradients(wrap)
                    attr = ig.attribute(xb_num2, baselines=torch.zeros_like(xb_num2),
                                        target=targets, n_steps=int(args.ig_steps), internal_batch_size=16)
                    num_attrib[i:i+args.batch_size] = attr.detach().cpu().numpy()
                    continue
                except Exception:
                    wrap_cpu = NumOnlyWrapper(mdl.to("cpu"), xb_cat.detach().cpu(), False, torch.float32).eval()
                    ig_cpu = IntegratedGradients(wrap_cpu)
                    xb_num_cpu = xb_num.detach().cpu().requires_grad_(True)
                    attr = ig_cpu.attribute(xb_num_cpu, baselines=torch.zeros_like(xb_num_cpu),
                                            target=targets.cpu(), n_steps=int(args.ig_steps), internal_batch_size=int(args.micro_batch_size))
                    num_attrib[i:i+args.batch_size] = attr.detach().cpu().numpy()
                    mdl.to(device)
    else:
        num_attrib = np.zeros((len(df), 0), dtype=np.float32)

    # 合并/缩放 → scores
    feat_names = list(cat_cols) + list(num_cols)
    cat_score = zscore_per_col(cat_deltas)
    num_score = zscore_per_col(num_attrib)
    scores = np.concatenate([cat_score, num_score], axis=1) if num_score.size else cat_score

    # ===== 导出：宽表 =====
    meta = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "pred_class": [classes[i] for i in pred_idx],
        "base_prob": probs[np.arange(len(df)), pred_idx]
    })
    if args.id_cols:
        keep = [c.strip() for c in args.id_cols.split(",") if c.strip()]
        if keep:
            meta = meta.merge(df[keep].reset_index().rename(columns={"index":"row_index"}), on="row_index", how="left")

    scores_df = pd.DataFrame(scores, columns=feat_names)
    wide_df = pd.concat([meta, scores_df], axis=1)
    wide_path = os.path.join(args.output_dir, "scores_wide.csv")
    wide_df.to_csv(wide_path, index=False)

    # ===== 导出：长表（可选，便于统计/绘图）=====
    long_df = scores_df.copy()
    long_df.insert(0, "row_index", np.arange(len(df)))
    long_df = long_df.melt(id_vars="row_index", var_name="feature", value_name="score")
    long_path = os.path.join(args.output_dir, "scores_long.csv")
    long_df.to_csv(long_path, index=False)

    # ===== 可选：保存原始矩阵 =====
    if args.save_raw:
        np.save(os.path.join(args.output_dir, "raw_cat_deltas.npy"), cat_deltas)
        np.save(os.path.join(args.output_dir, "raw_num_ig.npy"), num_attrib)
        np.save(os.path.join(args.output_dir, "scores.npy"), scores)

    # 同时保留全局重要性（|score| 的均值排名）
    glob = np.mean(np.abs(scores), axis=0) if scores.size else np.array([])
    pd.DataFrame({"feature": feat_names, "mean_abs_score": glob}) \
      .sort_values("mean_abs_score", ascending=False) \
      .to_csv(os.path.join(args.output_dir, "global_importance_fast.csv"), index=False)

    print("Saved:\n ", wide_path, "\n ", long_path,
          "\n ", os.path.join(args.output_dir, "global_importance_fast.csv"),
          ("\n  raw matrices saved." if args.save_raw else ""))

if __name__ == "__main__":
    main()

