
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, math, argparse, random
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_squared_error
import yaml

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MultiBandDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str, label_cols: List[str], path_cols: List[str],
                 img_size: int = 224, augment: bool = True):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.label_cols = label_cols
        self.path_cols = path_cols

        if augment:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def _load_gray(self, rel_path: str) -> Image.Image:
        path = os.path.join(self.image_root, rel_path) if self.image_root else rel_path
        img = Image.open(path).convert("L")
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgs = [self._load_gray(row[c]) for c in self.path_cols]
        tens = []
        for g in imgs:
            g3 = Image.merge("RGB", (g, g, g))
            tens.append(self.tf(g3))
        x = torch.stack(tens, dim=0).mean(dim=0)  # [3,H,W]
        y = torch.tensor([row[c] for c in self.label_cols], dtype=torch.float32)
        sid = str(row.get("sample_id", idx))
        return x, y, sid

class ViTRegressor(nn.Module):
    def __init__(self, backbone_name: str = "deit_tiny_patch16_224",
                 out_dim: int = 5, pretrained: bool = True,
                 freeze_backbone: bool = True, partial_ft_last_block: bool = False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        embed_dim = self.backbone.num_features
        self.head = nn.Linear(embed_dim, out_dim)
        for p in self.backbone.parameters():
            p.requires_grad = False
        if partial_ft_last_block:
            if hasattr(self.backbone, "blocks"):
                for p in self.backbone.blocks[-1].parameters(): p.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters(): p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)
        return out, feats

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))

@torch.no_grad()
def evaluate(model, loader, device, label_dim: int):
    model.eval()
    ys, ps = [], []
    for x, y, _ in loader:
        x = x.to(device); y = y.to(device)
        pred, _ = model(x)
        ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
    ys = np.concatenate(ys, axis=0); ps = np.concatenate(ps, axis=0)
    rmse_each = [rmse(ys[:,i], ps[:,i]) for i in range(label_dim)]
    return float(np.mean(rmse_each)), rmse_each

@torch.no_grad()
def predict_and_dump(model, loader, device, out_csv: str, label_cols: List[str]):
    model.eval()
    rows = []
    for x, y, sid in loader:
        x = x.to(device)
        pred, feats = model(x)
        pred = pred.detach().cpu().numpy()
        y = y.numpy()
        for i in range(len(sid)):
            row = {"sample_id": sid[i]}
            for j, lc in enumerate(label_cols):
                row[f"y_true_{lc}"] = float(y[i, j])
                row[f"y_pred_{lc}"] = float(pred[i, j])
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

@torch.no_grad()
def dump_embeddings(model, loader, device, out_parquet: str, label_cols: List[str]):
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception:
        print("[WARN] pyarrow not installed; skipping embedding parquet dump.")
        return
    model.eval()
    recs = []
    for x, y, sid in loader:
        x = x.to(device)
        _, feats = model(x)
        feats = feats.detach().cpu().numpy()
        y = y.numpy()
        for i in range(len(sid)):
            rec = {"sample_id": sid[i]}
            rec.update({f"f{k}": float(feats[i, k]) for k in range(feats.shape[1])})
            for j, lc in enumerate(label_cols):
                rec[f"y_{lc}"] = float(y[i, j])
            recs.append(rec)
    pd.DataFrame(recs).to_parquet(out_parquet)

def make_loaders(df_tr, df_va, df_te, cfg, label_cols, path_cols):
    ds_tr = MultiBandDataset(df_tr, cfg["data"]["image_root"], label_cols, path_cols, cfg["data"]["img_size"], augment=True)
    ds_va = MultiBandDataset(df_va, cfg["data"]["image_root"], label_cols, path_cols, cfg["data"]["img_size"], augment=False)
    ds_te = MultiBandDataset(df_te, cfg["data"]["image_root"], label_cols, path_cols, cfg["data"]["img_size"], augment=False)

    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True,
                       num_workers=cfg["train"]["workers"], pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=cfg["train"]["batch_size"], shuffle=False,
                       num_workers=cfg["train"]["workers"], pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=cfg["train"]["batch_size"], shuffle=False,
                       num_workers=cfg["train"]["workers"], pin_memory=True)
    return dl_tr, dl_va, dl_te

def split_data(df: pd.DataFrame, cfg: Dict[str, Any]):
    split_cfg = cfg["split"]
    if split_cfg.get("use_split_column", False) and ("split" in df.columns):
        df_tr = df[df["split"] == "train"].copy()
        df_va = df[df["split"] == "val"].copy()
        df_te = df[df["split"] == "test"].copy()
        return df_tr, df_va, df_te

    group_col = split_cfg.get("group_col", "group_id")
    if group_col not in df.columns:
        df[group_col] = df["sample_id"].astype(str)

    groups = df[group_col].astype(str)
    unique_groups = groups.drop_duplicates().values
    rng = np.random.RandomState(cfg["seed"])
    rng.shuffle(unique_groups)
    n_test_groups = max(1, int(len(unique_groups) * split_cfg.get("test_ratio", 0.2)))
    test_groups = set(unique_groups[:n_test_groups])
    is_test = groups.isin(test_groups)
    df_te = df[is_test].copy()
    df_tv = df[~is_test].copy()

    groups_tv = df_tv[group_col].astype(str).values
    ug = pd.Series(groups_tv).drop_duplicates().values
    rng.shuffle(ug)
    n_val_groups = max(1, int(len(ug) * split_cfg.get("val_ratio", 0.2)))
    val_groups = set(ug[:n_val_groups])
    is_val = df_tv[group_col].astype(str).isin(val_groups)
    df_va = df_tv[is_val].copy()
    df_tr = df_tv[~is_val].copy()
    return df_tr, df_va, df_te

def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 2025))

    df = pd.read_csv(cfg["data"]["csv_path"])

    need_cols = ["sample_id"] + cfg["data"]["label_cols"] + cfg["data"]["path_cols"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if cfg["data"].get("require_all_paths", True):
        mask = df[cfg["data"]["path_cols"]].notna().all(axis=1) & (df[cfg["data"]["path_cols"]].astype(str).apply(lambda s: all(len(x.strip())>0 for x in s), axis=1))
        df = df[mask].copy()

    df_tr, df_va, df_te = split_data(df, cfg)

    label_cols = cfg["data"]["label_cols"]
    path_cols  = cfg["data"]["path_cols"]
    dl_tr, dl_va, dl_te = make_loaders(df_tr, df_va, df_te, cfg, label_cols, path_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTRegressor(backbone_name=cfg["model"]["backbone"],
                         out_dim=len(label_cols),
                         pretrained=True,
                         freeze_backbone=not cfg["train"].get("partial_ft", False),
                         partial_ft_last_block=cfg["train"].get("partial_ft", False)).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    criterion = nn.SmoothL1Loss(beta=cfg["train"].get("huber_beta", 0.5))

    best = {"rmse_mean": 1e9, "epoch": -1}
    out_dir = cfg["train"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        tr_losses = []
        for x, y, _ in dl_tr:
            x = x.to(device); y = y.to(device)
            pred, _ = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        val_rmse_mean, val_rmse_each = evaluate(model, dl_va, device, len(label_cols))
        tr_loss = float(np.mean(tr_losses)) if tr_losses else None
        print(f"[E{epoch:02d}] train_loss={tr_loss:.4f}  val_RMSE(each)={np.round(val_rmse_each,3)}  mean={val_rmse_mean:.4f}")

        if val_rmse_mean < best["rmse_mean"]:
            best.update({"rmse_mean": val_rmse_mean, "rmse_each": val_rmse_each, "epoch": epoch})
            torch.save({"model": model.state_dict(), "cfg": cfg, "best": best},
                       os.path.join(out_dir, "best_model.pt"))

    state = torch.load(os.path.join(out_dir, "best_model.pt"), map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device); model.eval()

    test_rmse_mean, test_rmse_each = evaluate(model, dl_te, device, len(label_cols))
    print(f"[TEST] RMSE(each)={np.round(test_rmse_each,3)}  mean={test_rmse_mean:.4f}")

    preds_csv = os.path.join(out_dir, "test_predictions.csv")
    predict_and_dump(model, dl_te, device, preds_csv, label_cols)

    if cfg.get("export", {}).get("dump_test_embeddings", False):
        emb_parq = os.path.join(out_dir, "test_embeddings.parquet")
        dump_embeddings(model, dl_te, device, emb_parq, label_cols)

    metrics = {
        "best_val_rmse_mean": best["rmse_mean"],
        "best_val_rmse_each": best["rmse_each"],
        "best_epoch": best["epoch"],
        "test_rmse_mean": test_rmse_mean,
        "test_rmse_each": test_rmse_each,
        "n_train": len(df_tr), "n_val": len(df_va), "n_test": len(df_te)
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
