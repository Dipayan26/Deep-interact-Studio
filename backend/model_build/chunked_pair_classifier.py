"""Shared chunked two-sided interaction trainer for DTPI, RPI, and PDI."""

import json
from copy import deepcopy

import numpy as np
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

from model_build.sequence_models import FlexiblePairSequenceModel, _safe, trainable_parameter_count


def chunk_mask(chunks: torch.Tensor) -> torch.Tensor:
    return chunks.float().abs().sum(dim=-1).gt(0)


class ChunkedPairDataset(Dataset):
    def __init__(self, rows: list, left_dict: dict, right_dict: dict, name: str):
        self.samples = []
        skipped = 0
        for left_key, right_key, label in rows:
            left = left_dict.get(left_key)
            right = right_dict.get(right_key)
            if left is None or right is None:
                skipped += 1
                continue
            left = left.float()
            right = right.float()
            self.samples.append((left, right, chunk_mask(left), chunk_mask(right), float(label)))
        if skipped:
            print(f"[{name}] Skipped {skipped} pairs (missing chunked embeddings)", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        left, right, left_mask, right_mask, label = self.samples[idx]
        return (
            left.float(),
            right.float(),
            left_mask.bool(),
            right_mask.bool(),
            torch.tensor(label, dtype=torch.float),
        )


def _eval_model(model, data_loader, criterion, device):
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    probs_all, true_all = [], []
    with torch.inference_mode():
        for left, right, left_mask, right_mask, y in data_loader:
            left = left.to(device)
            right = right.to(device)
            left_mask = left_mask.to(device)
            right_mask = right_mask.to(device)
            y = y.to(device)
            logits = model(left, right, left_mask, right_mask)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            loss_total += loss.item() * len(y)
            correct += (preds == y.cpu().numpy().astype(int)).sum()
            total += len(y)
            probs_all.extend(probs.tolist())
            true_all.extend(y.cpu().numpy().tolist())
    return loss_total, correct, total, probs_all, true_all


def train_chunked_pair_classifier(
    rows: list,
    train_indices: list,
    val_indices: list,
    left_dict: dict,
    right_dict: dict,
    hyperparams: dict,
    metrics_path: str,
    model_path: str,
    task_type: str,
    left_dim_key: str,
    right_dim_key: str,
    left_dim: int,
    right_dim: int,
) -> dict:
    layer_configs = hyperparams.get("layer_configs", [
        {"type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3, "batchnorm": False},
        {"type": "linear", "hidden_dim": 64, "activation": "relu", "dropout": 0.2, "batchnorm": False},
    ])
    epochs = int(hyperparams.get("epochs", 30))
    lr = float(hyperparams.get("learning_rate", 0.001))
    batch_size = int(hyperparams.get("batch_size", 64))
    model_dim = int(hyperparams.get("chunk_model_dim", max(left_dim, right_dim)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_ds = ChunkedPairDataset([rows[i] for i in train_indices], left_dict, right_dict, task_type)
    va_ds = ChunkedPairDataset([rows[i] for i in val_indices], left_dict, right_dict, task_type)
    _pin = device == "cuda"
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=_pin)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=_pin)

    model = FlexiblePairSequenceModel(left_dim, right_dim, model_dim, layer_configs).to(device)
    actual_trainable_params = trainable_parameter_count(model)

    labels = [int(r[2]) for r in rows]
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
    no_improve = 0

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    va_probs, va_true = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for left, right, left_mask, right_mask, y in tr_dl:
            left = left.to(device)
            right = right.to(device)
            left_mask = left_mask.to(device)
            right_mask = right_mask.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(left, right, left_mask, right_mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            preds = (torch.sigmoid(logits) >= 0.5).long()
            tr_loss += loss.item() * len(y)
            tr_correct += (preds == y.long()).sum().item()
            tr_total += len(y)

        va_loss, va_correct, va_total, va_probs, va_true = _eval_model(model, va_dl, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(_safe(tr_loss / max(tr_total, 1)))
        history["val_loss"].append(_safe(va_loss / max(va_total, 1)))
        history["train_acc"].append(_safe(tr_correct / max(tr_total, 1)))
        history["val_acc"].append(_safe(va_correct / max(va_total, 1)))

        snapshot = {
            "status": "training",
            "epoch": epoch,
            "total_epochs": epochs,
            "history": history,
        }
        with open(metrics_path, "w") as f:
            json.dump(snapshot, f)

        current_val_loss = history["val_loss"][-1]
        current_val_loss = current_val_loss if current_val_loss is not None else float("inf")
        print(
            f"[{task_type} chunked epoch {epoch:03d}/{epochs}] "
            f"train_loss={history['train_loss'][-1]} val_loss={current_val_loss} "
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
                with open(metrics_path, "w") as f:
                    json.dump({**snapshot, "total_epochs": epoch, "early_stopped": True}, f)
                break

    model.load_state_dict(best_state)
    _, _, _, va_probs, va_true = _eval_model(model, va_dl, criterion, device)

    va_probs_arr = np.array(va_probs, dtype=float)
    va_true_arr = np.array(va_true, dtype=int)
    preds_arr = (va_probs_arr >= 0.5).astype(int)

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
            "recall": [_safe(v) for v in pr_rec[idx200_pr].tolist()],
        }
    except Exception:
        pr_data = None
    try:
        counts, bin_edges = np.histogram(va_probs_arr, bins=20, range=(0.0, 1.0))
        prob_hist = {"counts": counts.tolist(), "bins": [_safe(v) for v in bin_edges.tolist()]}
    except Exception:
        prob_hist = None

    final_metrics = {
        "status": "completed",
        "epoch": epoch,
        "total_epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": _safe(best_val_loss),
        "history": history,
        "final": {
            "val_acc": _safe(acc),
            "auroc": auroc,
            "ap": ap,
            "precision": _safe(prec),
            "recall": _safe(rec),
            "f1": _safe(f1),
            "trainable_params": actual_trainable_params,
        },
        "confusion_matrix": cm,
        "roc_curve": roc_data,
        "pr_curve": pr_data,
        "prob_hist": prob_hist,
        "val_probs": [_safe(v) for v in va_probs_arr.tolist()],
        "val_labels": va_true_arr.tolist(),
    }
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f)

    torch.save(
        {
            "model_state": model.state_dict(),
            "hyperparams": {**hyperparams, "embedding_representation": "chunked"},
            "input_dim": model_dim,
            "chunk_model_dim": model_dim,
            "layer_configs": layer_configs,
            left_dim_key: left_dim,
            right_dim_key: right_dim,
            "left_dim": left_dim,
            "right_dim": right_dim,
            "task_type": task_type,
            "embedding_representation": "chunked",
            "best_epoch": best_epoch,
            "best_val_loss": _safe(best_val_loss),
            "trainable_params": actual_trainable_params,
        },
        model_path,
    )
    print(f"[{task_type} chunked train] model saved → {model_path}", flush=True)
    return final_metrics
