"""
MLP-based PPI classifier on top of ESM2 pair embeddings.
"""

import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ESM2_DIM = 480  # esm2_t12_35M_UR50D hidden size

import math

def _safe(v):
    """Convert numpy scalars → Python float; replace NaN/Inf → None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Pair representation
# ---------------------------------------------------------------------------

def make_pair_vector(eA: torch.Tensor, eB: torch.Tensor, method: str) -> torch.Tensor:
    """Combine two protein embeddings into a single pair vector."""
    if method == "concat":
        return torch.cat([eA, eB], dim=-1)
    elif method == "product":
        return eA * eB
    elif method == "diff":
        return torch.abs(eA - eB)
    else:  # "all"  — concat + product + diff  (most common in literature)
        return torch.cat([eA, eB, eA * eB, torch.abs(eA - eB)], dim=-1)


def get_input_dim(method: str) -> int:
    if method in ("product", "diff"):
        return ESM2_DIM
    elif method == "concat":
        return ESM2_DIM * 2
    else:
        return ESM2_DIM * 4   # "all"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PPIClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PPIDataset(Dataset):
    def __init__(self, pairs, labels, embedding_dict: dict, method: str):
        self.samples = []
        skipped = 0
        for (seqA, seqB), label in zip(pairs, labels):
            eA = embedding_dict.get(seqA)
            eB = embedding_dict.get(seqB)
            if eA is None or eB is None:
                skipped += 1
                continue
            vec = make_pair_vector(eA, eB, method)
            self.samples.append((vec, int(label)))
        if skipped:
            print(f"[PPIDataset] Skipped {skipped} pairs (missing embeddings)", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vec, label = self.samples[idx]
        return vec.float(), torch.tensor(label, dtype=torch.long)


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
    Full training loop. Writes per-epoch metrics to metrics_path (JSON)
    so the frontend can poll it. Returns the final metrics dict.

    Expected df columns: proteinA, proteinB, label (0/1)
    """

    method     = hyperparams.get("pair_method", "all")
    hidden_dim = int(hyperparams.get("hidden_dim", 256))
    num_layers = int(hyperparams.get("num_layers", 2))
    epochs     = int(hyperparams.get("epochs", 30))
    lr         = float(hyperparams.get("learning_rate", 0.001))
    batch_size = int(hyperparams.get("batch_size", 64))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  method={method}  epochs={epochs}", flush=True)

    pairs  = list(zip(df["proteinA"].str.upper(), df["proteinB"].str.upper()))
    labels = df["label"].astype(int).tolist()

    # 80/20 stratified split
    idx = list(range(len(pairs)))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)

    tr_ds = PPIDataset([pairs[i] for i in tr_idx], [labels[i] for i in tr_idx], embedding_dict, method)
    va_ds = PPIDataset([pairs[i] for i in va_idx], [labels[i] for i in va_idx], embedding_dict, method)

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    input_dim = get_input_dim(method)
    model     = PPIClassifier(input_dim, hidden_dim, num_layers).to(device)

    # class-weighted loss to handle imbalance
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        w = torch.tensor([n_pos / len(labels), n_neg / len(labels)], dtype=torch.float).to(device)
    else:
        w = None
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    early_stop_patience = int(hyperparams.get("early_stop_patience", 0))  # 0 = disabled
    best_val_loss = float("inf")
    no_improve    = 0

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # ── train ────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for X, y in tr_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * len(y)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total   += len(y)

        # ── validate ──────────────────────────────────────────────────────
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        va_probs, va_true = [], []
        with torch.no_grad():
            for X, y in va_dl:
                X, y = X.to(device), y.to(device)
                out  = model(X)
                loss = criterion(out, y)
                va_loss    += loss.item() * len(y)
                va_correct += (out.argmax(1) == y).sum().item()
                va_total   += len(y)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                va_probs.extend(probs.tolist())
                va_true.extend(y.cpu().numpy().tolist())

        history["epoch"].append(epoch)
        history["train_loss"].append(_safe(tr_loss / max(tr_total, 1)))
        history["val_loss"].append(_safe(va_loss   / max(va_total, 1)))
        history["train_acc"].append(_safe(tr_correct / max(tr_total, 1)))
        history["val_acc"].append(_safe(va_correct   / max(va_total, 1)))

        # write after every epoch so frontend can poll
        snapshot = {
            "status": "training",
            "epoch": epoch,
            "total_epochs": epochs,
            "history": history,
        }
        with open(metrics_path, "w") as f:
            json.dump(snapshot, f)

        current_val_loss = history["val_loss"][-1] or float("inf")
        print(
            f"[epoch {epoch:03d}/{epochs}] "
            f"train_loss={history['train_loss'][-1]}  "
            f"val_loss={current_val_loss}  "
            f"val_acc={history['val_acc'][-1]}",
            flush=True,
        )

        # ── early stopping ────────────────────────────────────────────────
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
                    # mark total_epochs as actual epoch reached so progress bar fills
                    with open(metrics_path, "w") as f:
                        json.dump({**snapshot, "total_epochs": epoch,
                                   "early_stopped": True}, f)
                    break

    # ── final metrics ─────────────────────────────────────────────────────
    try:
        auroc = _safe(roc_auc_score(va_true, va_probs))
    except Exception:
        auroc = None

    preds = [1 if p >= 0.5 else 0 for p in va_probs]
    prec, rec, f1, _ = precision_recall_fscore_support(
        va_true, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(va_true, preds)

    final_metrics = {
        "status": "completed",
        "epoch": epochs,
        "total_epochs": epochs,
        "history": history,
        "final": {
            "val_acc":   _safe(acc),
            "auroc":     auroc,
            "precision": _safe(prec),
            "recall":    _safe(rec),
            "f1":        _safe(f1),
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f)

    # ── save model ────────────────────────────────────────────────────────
    torch.save(
        {
            "model_state": model.state_dict(),
            "hyperparams": hyperparams,
            "input_dim":   input_dim,
            "hidden_dim":  hidden_dim,
            "num_layers":  num_layers,
        },
        model_path,
    )
    print(f"[train] model saved → {model_path}", flush=True)

    return final_metrics
