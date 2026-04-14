"""
Flexible PPI classifier supporting Linear, CNN1D, BiLSTM, GRU, Transformer,
and Residual layer types on top of ESM2 pair embeddings.
"""

import json
import math
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe(v):
    """Convert numpy scalars → Python float; replace NaN/Inf → None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _get_act(name: str) -> nn.Module:
    """Return a fresh activation module for the given name."""
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    if name == "silu":
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    return nn.ReLU()


# ---------------------------------------------------------------------------
# Layer blocks
# ---------------------------------------------------------------------------

class _LinearBlock(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        hidden = int(cfg.get("hidden_dim", 256))
        act    = cfg.get("activation", "relu")
        drop   = float(cfg.get("dropout", 0.3))
        bn     = bool(cfg.get("batchnorm", False))

        layers = [nn.Linear(in_dim, hidden)]
        if bn:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(_get_act(act))
        layers.append(nn.Dropout(drop))
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CNN1DBlock(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        out_ch = int(cfg.get("out_channels", 64))
        k      = int(cfg.get("kernel_size", 3))
        act    = cfg.get("activation", "relu")
        drop   = float(cfg.get("dropout", 0.3))

        self.conv = nn.Conv1d(1, out_ch, kernel_size=k, padding=k // 2)
        self.act  = _get_act(act)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(drop)
        self.out_dim = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) → (B, 1, in_dim)
        x = x.unsqueeze(1)
        x = self.conv(x)          # (B, out_ch, in_dim)
        x = self.act(x)
        x = self.pool(x)          # (B, out_ch, 1)
        x = x.squeeze(-1)         # (B, out_ch)
        x = self.drop(x)
        return x


class _BiLSTMBlock(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        hidden     = int(cfg.get("hidden_size", 128))
        num_layers = int(cfg.get("num_layers", 1))
        drop       = float(cfg.get("dropout", 0.3))

        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=drop if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(drop)
        self.out_dim = 2 * hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) → (B, 1, in_dim)
        x = x.unsqueeze(1)
        out, (h, _) = self.lstm(x)
        # Take last forward + backward hidden state
        x = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 2*hidden)
        x = self.drop(x)
        return x


class _GRUBlock(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        hidden        = int(cfg.get("hidden_size", 128))
        num_layers    = int(cfg.get("num_layers", 1))
        bidirectional = bool(cfg.get("bidirectional", True))
        drop          = float(cfg.get("dropout", 0.3))

        self.gru = nn.GRU(
            input_size=in_dim, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=drop if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(drop)
        self.out_dim = 2 * hidden if bidirectional else hidden
        self._bidir = bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        _, h = self.gru(x)
        if self._bidir:
            x = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            x = h[-1]
        x = self.drop(x)
        return x


class _TransformerBlock(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        d_model    = int(cfg.get("d_model", 256))
        nhead      = int(cfg.get("nhead", 4))
        num_layers = int(cfg.get("num_layers", 2))
        dim_ff     = int(cfg.get("dim_feedforward", d_model * 2))
        drop       = float(cfg.get("dropout", 0.1))

        # Ensure d_model % nhead == 0 by reducing nhead
        while nhead > 1 and d_model % nhead != 0:
            nhead -= 1

        self.proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=drop, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(drop)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)       # (B, d_model)
        x = x.unsqueeze(1)     # (B, 1, d_model)
        x = self.encoder(x)    # (B, 1, d_model)
        x = x.squeeze(1)       # (B, d_model)
        x = self.drop(x)
        return x


class _ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        hidden = int(cfg.get("hidden_dim", 256))
        act    = cfg.get("activation", "relu")
        drop   = float(cfg.get("dropout", 0.3))
        bn     = bool(cfg.get("batchnorm", False))

        layers = [nn.Linear(in_dim, hidden)]
        if bn:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(_get_act(act))
        layers.append(nn.Dropout(drop))
        layers.append(nn.Linear(hidden, in_dim))
        self.net  = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(in_dim)
        self.out_dim = in_dim  # residual block preserves dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


# ---------------------------------------------------------------------------
# Builder registry
# ---------------------------------------------------------------------------

def _build_layer(layer_type: str, in_dim: int, cfg: dict) -> nn.Module:
    lt = layer_type.lower()
    if lt == "linear":
        return _LinearBlock(in_dim, cfg)
    if lt == "cnn1d":
        return _CNN1DBlock(in_dim, cfg)
    if lt == "bilstm":
        return _BiLSTMBlock(in_dim, cfg)
    if lt == "gru":
        return _GRUBlock(in_dim, cfg)
    if lt == "transformer":
        return _TransformerBlock(in_dim, cfg)
    if lt == "residual":
        return _ResidualBlock(in_dim, cfg)
    raise ValueError(f"Unknown layer type: '{layer_type}'")


# ---------------------------------------------------------------------------
# Flexible model
# ---------------------------------------------------------------------------

class FlexiblePPIModel(nn.Module):
    def __init__(self, input_dim: int, layer_configs: list):
        super().__init__()

        if not layer_configs:
            layer_configs = [
                {"type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3},
                {"type": "linear", "hidden_dim": 64,  "activation": "relu", "dropout": 0.2},
            ]

        blocks = []
        cur_dim = input_dim
        for cfg in layer_configs:
            block = _build_layer(cfg["type"], cur_dim, cfg)
            blocks.append(block)
            cur_dim = block.out_dim

        self.blocks    = nn.ModuleList(blocks)
        self.output    = nn.Linear(cur_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _pair_vec(eA: torch.Tensor, eB: torch.Tensor, mode: str) -> torch.Tensor:
    eA = eA.float()
    eB = eB.float()
    if mode == "product":
        return eA * eB
    if mode == "diff":
        return (eA - eB).abs()
    if mode == "all":
        return torch.cat([eA, eB, eA * eB, (eA - eB).abs()], dim=-1)
    return torch.cat([eA, eB], dim=-1)  # concat (default)


def _pair_input_dim(esm_dim: int, mode: str) -> int:
    if mode in ("product", "diff"):
        return esm_dim
    if mode == "all":
        return 4 * esm_dim
    return 2 * esm_dim


class PPIDataset(Dataset):
    def __init__(self, pairs, labels, embedding_dict: dict, pair_mode: str = "concat"):
        self.samples = []
        skipped = 0
        for (seqA, seqB), label in zip(pairs, labels):
            eA = embedding_dict.get(seqA)
            eB = embedding_dict.get(seqB)
            if eA is None or eB is None:
                skipped += 1
                continue
            vec = _pair_vec(eA, eB, pair_mode)
            self.samples.append((vec, float(label)))
        if skipped:
            print(f"[PPIDataset] Skipped {skipped} pairs (missing embeddings)", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vec, label = self.samples[idx]
        return vec.float(), torch.tensor(label, dtype=torch.float)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(
    df: pd.DataFrame,
    embedding_dict: dict,
    hyperparams: dict,
    metrics_path: str,
    model_path: str,
) -> dict:
    """
    Full training loop with flexible architecture.
    Writes per-epoch metrics to metrics_path (JSON) for frontend polling.
    Returns the final metrics dict.

    Expected df columns: proteinA, proteinB, label (0/1)
    """

    layer_configs = hyperparams.get("layer_configs", [
        {"type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3, "batchnorm": False},
        {"type": "linear", "hidden_dim": 64,  "activation": "relu", "dropout": 0.2, "batchnorm": False},
    ])
    esm_dim    = int(hyperparams.get("esm_dim", 480))
    epochs     = int(hyperparams.get("epochs", 30))
    lr         = float(hyperparams.get("learning_rate", 0.001))
    batch_size = int(hyperparams.get("batch_size", 64))
    pair_mode  = str(hyperparams.get("pair_representation", "concat")).lower()
    if pair_mode not in ("concat", "product", "diff", "all"):
        print(f"[train] unknown pair_representation={pair_mode!r}; falling back to 'concat'", flush=True)
        pair_mode = "concat"
    input_dim  = _pair_input_dim(esm_dim, pair_mode)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  esm_dim={esm_dim}  pair_mode={pair_mode}  input_dim={input_dim}  epochs={epochs}", flush=True)

    pairs  = list(zip(
        df["proteinA"].astype(str).str.strip().str.upper(),
        df["proteinB"].astype(str).str.strip().str.upper(),
    ))
    labels = df["label"].astype(int).tolist()

    # 80/20 stratified split
    idx = list(range(len(pairs)))
    try:
        tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)
    except ValueError:
        # Too few samples for stratified split — fall back to random
        tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=42)

    tr_ds = PPIDataset([pairs[i] for i in tr_idx], [labels[i] for i in tr_idx], embedding_dict, pair_mode)
    va_ds = PPIDataset([pairs[i] for i in va_idx], [labels[i] for i in va_idx], embedding_dict, pair_mode)

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = FlexiblePPIModel(input_dim, layer_configs).to(device)

    # BCEWithLogitsLoss with pos_weight for class imbalance
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float).to(device)
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    early_stop_patience = int(hyperparams.get("early_stop_patience", 0))
    best_val_loss = float("inf")
    no_improve    = 0

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    va_probs, va_true = [], []  # Will be overwritten each epoch; kept for final metrics

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for X, y in tr_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            preds       = (torch.sigmoid(logits) >= 0.5).long()
            tr_loss    += loss.item() * len(y)
            tr_correct += (preds == y.long()).sum().item()
            tr_total   += len(y)

        # Validate
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        va_probs, va_true = [], []
        with torch.no_grad():
            for X, y in va_dl:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss   = criterion(logits, y)
                probs  = torch.sigmoid(logits).cpu().numpy()
                preds  = (probs >= 0.5).astype(int)
                va_loss    += loss.item() * len(y)
                va_correct += (preds == y.cpu().numpy().astype(int)).sum()
                va_total   += len(y)
                va_probs.extend(probs.tolist())
                va_true.extend(y.cpu().numpy().tolist())

        history["epoch"].append(epoch)
        history["train_loss"].append(_safe(tr_loss / max(tr_total, 1)))
        history["val_loss"].append(_safe(va_loss   / max(va_total, 1)))
        history["train_acc"].append(_safe(tr_correct / max(tr_total, 1)))
        history["val_acc"].append(_safe(va_correct   / max(va_total, 1)))

        snapshot = {
            "status":       "training",
            "epoch":        epoch,
            "total_epochs": epochs,
            "history":      history,
        }
        with open(metrics_path, "w") as f:
            json.dump(snapshot, f)

        _vl = history["val_loss"][-1]
        current_val_loss = _vl if _vl is not None else float("inf")
        print(
            f"[epoch {epoch:03d}/{epochs}] "
            f"train_loss={history['train_loss'][-1]}  "
            f"val_loss={current_val_loss}  "
            f"val_acc={history['val_acc'][-1]}",
            flush=True,
        )

        if early_stop_patience > 0:
            if current_val_loss < best_val_loss - 1e-4:
                best_val_loss = current_val_loss
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    print(
                        f"[early stop] No improvement for {early_stop_patience} epochs. "
                        f"Stopping at epoch {epoch}.",
                        flush=True,
                    )
                    with open(metrics_path, "w") as f:
                        json.dump({**snapshot, "total_epochs": epoch, "early_stopped": True}, f)
                    break

    # Final metrics
    va_probs_arr = np.array(va_probs, dtype=float)
    va_true_arr  = np.array(va_true,  dtype=int)
    preds_arr    = (va_probs_arr >= 0.5).astype(int)

    try:
        auroc = _safe(roc_auc_score(va_true_arr, va_probs_arr))
    except Exception:
        auroc = None

    try:
        ap = _safe(average_precision_score(va_true_arr, va_probs_arr))
    except Exception:
        ap = None

    prec, rec, f1, _ = precision_recall_fscore_support(
        va_true_arr, preds_arr, average="binary", zero_division=0
    )
    acc = accuracy_score(va_true_arr, preds_arr)

    # Confusion matrix
    try:
        cm = confusion_matrix(va_true_arr, preds_arr).tolist()  # [[TN,FP],[FN,TP]]
    except Exception:
        cm = None

    # ROC curve (down-sampled to 200 points)
    try:
        fpr_full, tpr_full, _ = roc_curve(va_true_arr, va_probs_arr)
        idx200 = np.linspace(0, len(fpr_full) - 1, min(200, len(fpr_full)), dtype=int)
        roc_data = {
            "fpr": [_safe(v) for v in fpr_full[idx200].tolist()],
            "tpr": [_safe(v) for v in tpr_full[idx200].tolist()],
        }
    except Exception:
        roc_data = None

    # Precision-Recall curve (down-sampled to 200 points)
    try:
        pr_prec, pr_rec, _ = precision_recall_curve(va_true_arr, va_probs_arr)
        idx200_pr = np.linspace(0, len(pr_prec) - 1, min(200, len(pr_prec)), dtype=int)
        pr_data = {
            "precision": [_safe(v) for v in pr_prec[idx200_pr].tolist()],
            "recall":    [_safe(v) for v in pr_rec[idx200_pr].tolist()],
        }
    except Exception:
        pr_data = None

    # Probability histogram (20 bins)
    try:
        counts, bin_edges = np.histogram(va_probs_arr, bins=20, range=(0.0, 1.0))
        prob_hist = {
            "counts": counts.tolist(),
            "bins":   [_safe(v) for v in bin_edges.tolist()],
        }
    except Exception:
        prob_hist = None

    final_metrics = {
        "status":       "completed",
        "epoch":        epoch,
        "total_epochs": epochs,
        "history":      history,
        "final": {
            "val_acc":   _safe(acc),
            "auroc":     auroc,
            "ap":        ap,
            "precision": _safe(prec),
            "recall":    _safe(rec),
            "f1":        _safe(f1),
        },
        "confusion_matrix": cm,
        "roc_curve":        roc_data,
        "pr_curve":         pr_data,
        "prob_hist":        prob_hist,
    }
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f)

    # Save checkpoint
    torch.save(
        {
            "model_state":  model.state_dict(),
            "hyperparams":  hyperparams,
            "input_dim":    input_dim,
            "layer_configs": layer_configs,
            "esm_dim":      esm_dim,
            "pair_mode":    pair_mode,
        },
        model_path,
    )
    print(f"[train] model saved → {model_path}", flush=True)

    return final_metrics
