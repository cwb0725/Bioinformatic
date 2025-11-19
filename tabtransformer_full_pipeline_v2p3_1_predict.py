#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TabTransformer inference (v2.3.1, PREDICT)
- Loads per-fold artifacts: preprocessor.json, cat_vocabs.json, (optional) scaler.pkl, model.pt
- Does NOT require --train_data anymore.
"""

import os, json, argparse, glob, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------- Minimal dataset/model --------------------------

class TabDataset(Dataset):
    def __init__(self, Xcat, Xnum):
        import numpy as _np, torch as _torch
        if isinstance(Xcat, _np.ndarray): Xcat = _np.ascontiguousarray(Xcat).copy()
        if isinstance(Xnum, _np.ndarray): Xnum = _np.ascontiguousarray(Xnum).copy()
        self.Xcat = _torch.tensor(Xcat, dtype=_torch.long) if Xcat.size else _torch.zeros((len(Xnum),0), dtype=_torch.long)
        self.Xnum = _torch.tensor(Xnum, dtype=_torch.float32) if Xnum.size else _torch.zeros((len(Xnum),0), dtype=_torch.float32)
    def __len__(self): return self.Xnum.shape[0] if self.Xnum.ndim==2 else self.Xcat.shape[0]
    def __getitem__(self, i): return self.Xcat[i], self.Xnum[i]

class TabTransformer(nn.Module):
    def __init__(self, cardinalities, num_dim, n_classes, d_model=256, n_heads=8, depth=4, p_drop=0.05):
        super().__init__()
        self.n_cat=len(cardinalities); self.n_num=num_dim; self.d_model=d_model
        self.cat_embeddings=nn.ModuleList([nn.Embedding(card, d_model) for card in cardinalities])
        self.col_embeddings=nn.Embedding(self.n_cat, d_model); self.cls=nn.Parameter(torch.zeros(1,1,d_model))
        enc_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True,
                    dim_feedforward=4*d_model, dropout=p_drop, activation="gelu", norm_first=False)
        self.encoder=nn.TransformerEncoder(enc_layer, num_layers=depth); self.post_norm=nn.LayerNorm(d_model)
        hid=max(64,d_model)
        self.num_tower=(nn.Sequential(nn.LayerNorm(num_dim) if num_dim>0 else nn.Identity(),
                                      nn.Linear(num_dim if num_dim>0 else 1, hid), nn.GELU(), nn.Dropout(p_drop), nn.Linear(hid, d_model))
                        if num_dim>0 else None)
        out_in=d_model*(2 if num_dim>0 else 1)
        self.classifier=nn.Sequential(nn.Linear(out_in, d_model), nn.GELU(), nn.Dropout(p_drop), nn.Linear(d_model, n_classes))
    def forward(self, xcat, xnum):
        B=xcat.size(0)
        if self.n_cat>0:
            toks=[]
            for j,emb in enumerate(self.cat_embeddings):
                t=emb(xcat[:,j]) + self.col_embeddings(torch.full((B,), j, device=xcat.device))
                toks.append(t.unsqueeze(1))
            cat_tokens=torch.cat(toks, dim=1); seq=torch.cat([self.cls.expand(B,1,-1), cat_tokens], dim=1)
        else:
            seq=self.cls.expand(B,1,-1)
        h=self.encoder(seq); h_cls=self.post_norm(h[:,0,:])
        if self.num_tower is not None and xnum is not None and xnum.shape[1]>0:
            h_num=self.num_tower(xnum); h_all=torch.cat([h_cls,h_num], dim=1)
        else: h_all=h_cls
        return self.classifier(h_all)

def load_json(p):
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def map_categoricals(df_in: pd.DataFrame, cat_cols, vocabs: dict) -> np.ndarray:
    """Map categorical columns using saved value->id vocab; unseen -> <RARE>/<UNK>"""
    X_list = []
    for col in cat_cols:
        vocab = vocabs.get(col, {"<UNK>":0})
        unk_id = vocab.get("<UNK>", 0)
        rare_id = vocab.get("<RARE>", unk_id)
        series = df_in[col] if col in df_in.columns else pd.Series([pd.NA]*len(df_in))
        vals = []
        for v in series.astype("object").tolist():
            key = "<NA>" if pd.isna(v) else str(v)
            vals.append(vocab.get(key, rare_id))
        X_list.append(np.array(vals, dtype=np.int64))
    return np.vstack(X_list).T if X_list else np.zeros((len(df_in),0), dtype=np.int64)

def maybe_scale_numeric(Xnum: np.ndarray, scaler_path: str):
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            return scaler.transform(Xnum.astype(np.float32, copy=False))
        except Exception as e:
            print(f"[WARN] Failed to apply scaler from {scaler_path}. Details: {e}")
    return Xnum.astype(np.float32, copy=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_outdir", type=str, required=True, help="Root dir containing fold_* subdirs")
    ap.add_argument("--input_csv", type=str, required=True, help="CSV to predict")
    ap.add_argument("--output_csv", type=str, required=True, help="Where to write predictions")
    ap.add_argument("--id_cols", type=str, default="", help="Comma-separated columns to carry to output")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16","bf16"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = (not args.no_amp) and (device=="cuda")
    amp_dtype = torch.float16 if args.amp_dtype=="fp16" else torch.bfloat16

    classes = load_json(os.path.join(args.model_outdir, "class_names.json"))["class_names"]
    n_classes = len(classes)

    fold_dirs = sorted([p for p in glob.glob(os.path.join(args.model_outdir, "fold_*")) if os.path.isdir(p)])
    if not fold_dirs:
        raise RuntimeError("No fold_* directories found under --model_outdir")

    df_in = pd.read_csv(args.input_csv)

    logits_sum = None
    for fdir in fold_dirs:
        pre = load_json(os.path.join(fdir, "preprocessor.json"))
        cat_cols = pre.get("cat_cols", [])
        num_cols = pre.get("num_cols", [])
        cardinalities = pre.get("cardinalities", [])
        num_dim = pre.get("numeric_dim", len(num_cols))

        # ensure columns exist
        for c in cat_cols:
            if c not in df_in.columns:
                df_in[c] = pd.NA
        for c in num_cols:
            if c not in df_in.columns:
                df_in[c] = np.nan

        # map cats using saved vocabs
        vocabs = load_json(os.path.join(fdir, "cat_vocabs.json"))
        Xcat = map_categoricals(df_in, cat_cols, vocabs)

        # numeric (pass-through + optional saved scaler)
        Xnum = df_in[num_cols].to_numpy(copy=True) if num_cols else np.zeros((len(df_in),0), dtype=np.float32)
        scaler_path = os.path.join(fdir, "scaler.pkl")
        Xnum = maybe_scale_numeric(Xnum, scaler_path)

        ds = TabDataset(Xcat, Xnum)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

        # build model with exact shapes
        d_model = 256; depth = 4; heads = 8; pdrop = 0.05
        try:
            tr_cfg = load_json(os.path.join(args.model_outdir, "train_config.json"))
            d_model = tr_cfg.get("d_model", d_model)
            depth = tr_cfg.get("depth", depth)
            heads = tr_cfg.get("n_heads", heads)
            pdrop = tr_cfg.get("dropout", pdrop)
        except Exception:
            pass

        mdl = TabTransformer(cardinalities=cardinalities, num_dim=num_dim, n_classes=n_classes,
                             d_model=d_model, n_heads=heads, depth=depth, p_drop=pdrop).to(device).eval()

        state = torch.load(os.path.join(fdir, "model.pt"), map_location=device, weights_only=True)
        mdl.load_state_dict(state)

        outs=[]
        with torch.no_grad():
            for xcat, xnum in dl:
                xcat = xcat.to(device); xnum = xnum.to(device)
                ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
                with ctx:
                    out = mdl(xcat, xnum)
                outs.append(out.detach().cpu())
        logits = torch.cat(outs, dim=0)
        logits_sum = logits if logits_sum is None else logits_sum + logits

    logits_mean = logits_sum / float(len(fold_dirs))
    probs = torch.softmax(logits_mean, dim=1).numpy()
    preds_idx = probs.argmax(axis=1)
    preds = [classes[i] for i in preds_idx]

    out_df = pd.DataFrame({"pred": preds})
    for i, name in enumerate(classes):
        out_df[f"prob_{name}"] = probs[:, i]

    if args.id_cols:
        keep = [c.strip() for c in args.id_cols.split(",") if c.strip()]
        for c in keep:
            if c in df_in.columns:
                out_df[c] = df_in[c].values
        cols = [c for c in keep if c in out_df.columns] + [c for c in out_df.columns if c not in keep]
        out_df = out_df[cols]

    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")

if __name__ == "__main__":
    main()
