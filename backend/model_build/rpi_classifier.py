"""
RNA-Protein Interaction (RPI) classifier training loop.
Input: concat(RNA-FM_emb, ESM2_emb)
Architecture: reuses FlexiblePPIModel from ppi_classifier.
Expected CSV columns: rna_sequence, protein_sequence, label
"""

import json
import math
from copy import deepcopy

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

from model_build.chunked_pair_classifier import train_chunked_pair_classifier
from model_build.ppi_classifier import FlexiblePPIModel
from model_build.sequence_models import _safe, _train_test_size


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RPIDataset(Dataset):
    """
    Each sample: concat(rna_embedding, esm_embedding) → binary label.
    """
    def __init__(self, rows: list, rna_dict: dict, esm_dict: dict):
        """
        rows     : list of (rna_seq_str, protein_seq_str, label_int)
        rna_dict : {rna_seq -> torch.Tensor}  (RNA-FM embeddings)
        esm_dict : {protein_seq -> torch.Tensor} (ESM2 embeddings)
        """
        self.samples = []
        skipped = 0
        for rna_seq, prot_seq, label in rows:
            e_rna  = rna_dict.get(rna_seq)
            e_prot = esm_dict.get(prot_seq)
            if e_rna is None or e_prot is None:
                skipped += 1
                continue
            vec = torch.cat([e_rna.float(), e_prot.float()], dim=-1)
            self.samples.append((vec, float(label)))
        if skipped:
            print(f"[RPIDataset] Skipped {skipped} pairs (missing embeddings)", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        vec, label = self.samples[idx]
        return vec.float(), torch.tensor(label, dtype=torch.float)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_rpi_classifier(
    df: pd.DataFrame,
    rna_dict: dict,
    esm_dict: dict,
    hyperparams: dict,
    metrics_path: str,
    model_path: str,
) -> dict:
    """
    Full RPI training loop with flexible architecture.
    Writes per-epoch metrics JSON for frontend polling.
    Returns final metrics dict.
    """
    layer_configs = hyperparams.get("layer_configs", [
        {"type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3, "batchnorm": False},
        {"type": "linear", "hidden_dim": 64,  "activation": "relu", "dropout": 0.2, "batchnorm": False},
    ])
    rna_dim    = int(hyperparams.get("rna_dim",   640))
    esm_dim    = int(hyperparams.get("esm_dim",   480))
    representation_mode = str(
        hyperparams.get("embedding_representation", hyperparams.get("representation_mode", "pooled"))
    ).lower()
    if representation_mode not in ("pooled", "chunked"):
        representation_mode = "pooled"
    input_dim = int(hyperparams.get("chunk_model_dim", max(rna_dim, esm_dim))) if representation_mode == "chunked" else rna_dim + esm_dim
    epochs     = int(hyperparams.get("epochs",    30))
    lr         = float(hyperparams.get("learning_rate", 0.001))
    batch_size = int(hyperparams.get("batch_size", 64))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[rpi_train] device={device}  mode={representation_mode}  rna_dim={rna_dim}  esm_dim={esm_dim}  "
        f"input_dim={input_dim}  epochs={epochs}",
        flush=True,
    )

    rows = list(zip(
        df["rna_sequence"].astype(str).str.strip().str.upper().str.replace("T", "U", regex=False),
        df["protein_sequence"].astype(str).str.strip().str.upper(),
        df["label"].astype(int),
    ))
    labels = [int(r[2]) for r in rows]

    test_size = _train_test_size(hyperparams)
    idx = list(range(len(rows)))
    try:
        tr_idx, va_idx = train_test_split(idx, test_size=test_size, stratify=labels, random_state=42)
    except ValueError:
        tr_idx, va_idx = train_test_split(idx, test_size=test_size, random_state=42)

    if representation_mode == "chunked":
        return train_chunked_pair_classifier(
            rows=rows,
            train_indices=tr_idx,
            val_indices=va_idx,
            left_dict=rna_dict,
            right_dict=esm_dict,
            hyperparams=hyperparams,
            metrics_path=metrics_path,
            model_path=model_path,
            task_type="rpi",
            left_dim_key="rna_dim",
            right_dim_key="esm_dim",
            left_dim=rna_dim,
            right_dim=esm_dim,
        )

    tr_ds = RPIDataset([rows[i] for i in tr_idx], rna_dict, esm_dict)
    va_ds = RPIDataset([rows[i] for i in va_idx], rna_dict, esm_dict)

    _pin = (device == "cuda")
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  drop_last=False, pin_memory=_pin)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=_pin)

    model = FlexiblePPIModel(input_dim, layer_configs).to(device)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pos_weight = (
        torch.tensor([n_neg / n_pos], dtype=torch.float).to(device)
        if n_pos > 0 and n_neg > 0
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    early_stop_patience = int(hyperparams.get("early_stop_patience", 0))
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    no_improve    = 0

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    va_probs: list = []
    va_true:  list = []

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
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

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        va_probs, va_true = [], []
        with torch.inference_mode():
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
            f"[rpi epoch {epoch:03d}/{epochs}] "
            f"train_loss={history['train_loss'][-1]}  "
            f"val_loss={current_val_loss}  "
            f"val_acc={history['val_acc'][-1]}",
            flush=True,
        )

        if current_val_loss < best_val_loss - 1e-4:
            best_val_loss = current_val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if early_stop_patience > 0 and no_improve >= early_stop_patience:
                print(
                    f"[rpi early stop] No improvement for {early_stop_patience} epochs. "
                    f"Stopping at epoch {epoch}.",
                    flush=True,
                )
                with open(metrics_path, "w") as f:
                    json.dump({**snapshot, "total_epochs": epoch, "early_stopped": True}, f)
                break

    model.load_state_dict(best_state)

    # ── Final metrics ─────────────────────────────────────────────────────────
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

    try:
        cm = confusion_matrix(va_true_arr, preds_arr).tolist()
    except Exception:
        cm = None

    try:
        fpr_full, tpr_full, _ = roc_curve(va_true_arr, va_probs_arr)
        idx200 = np.linspace(0, len(fpr_full) - 1, min(200, len(fpr_full)), dtype=int)
        roc_data = {
            "fpr": [_safe(v) for v in fpr_full[idx200].tolist()],
            "tpr": [_safe(v) for v in tpr_full[idx200].tolist()],
        }
    except Exception:
        roc_data = None

    try:
        pr_prec, pr_rec, _ = precision_recall_curve(va_true_arr, va_probs_arr)
        idx200_pr = np.linspace(0, len(pr_prec) - 1, min(200, len(pr_prec)), dtype=int)
        pr_data = {
            "precision": [_safe(v) for v in pr_prec[idx200_pr].tolist()],
            "recall":    [_safe(v) for v in pr_rec[idx200_pr].tolist()],
        }
    except Exception:
        pr_data = None

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
        "best_epoch":   best_epoch,
        "best_val_loss": _safe(best_val_loss),
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
        "val_probs":        [_safe(v) for v in va_probs_arr.tolist()],
        "val_labels":       va_true_arr.tolist(),
    }
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f)

    torch.save(
        {
            "model_state":   model.state_dict(),
            "hyperparams":   hyperparams,
            "input_dim":     input_dim,
            "layer_configs": layer_configs,
            "rna_dim":       rna_dim,
            "esm_dim":       esm_dim,
            "task_type":     "rpi",
            "embedding_representation": representation_mode,
            "best_epoch":    best_epoch,
            "best_val_loss": _safe(best_val_loss),
        },
        model_path,
    )
    print(f"[rpi_train] model saved → {model_path}", flush=True)

    return final_metrics
