"""
Inference utilities: load a trained FlexiblePPIModel and score new pairs.

Handles both the current block-based architecture and legacy flat-net checkpoints,
including old models with a 2-class output head.
"""

import re

import torch
import torch.nn as nn
import pandas as pd
from model_build.ppi_classifier import FlexiblePPIModel
from model_build.sequence_models import FlexiblePPISequenceModel


# ---------------------------------------------------------------------------
# Legacy checkpoint helpers
# ---------------------------------------------------------------------------

def _remap_legacy_state_dict(sd: dict) -> dict:
    """
    Convert old flat-net checkpoints (net.0.*, net.3.*, …) to the current
    block-based format (blocks.N.net.0.*, output.*).

    Old architecture stored every hidden layer as 3 Sequential entries:
      net.{3k}.weight / .bias  →  blocks.{k}.net.0.weight / .bias
    The final Linear is the output head:
      net.{3*N}.weight / .bias →  output.weight / .bias
    """
    if not any(k.startswith("net.") for k in sd):
        return sd  # already new-format

    indices = sorted({
        int(m.group(1))
        for k in sd
        if (m := re.match(r"net\.(\d+)\.weight$", k))
    })

    new_sd = {}
    for block_i, net_i in enumerate(indices[:-1]):
        new_sd[f"blocks.{block_i}.net.0.weight"] = sd[f"net.{net_i}.weight"]
        new_sd[f"blocks.{block_i}.net.0.bias"]   = sd[f"net.{net_i}.bias"]

    last = indices[-1]
    new_sd["output.weight"] = sd[f"net.{last}.weight"]
    new_sd["output.bias"]   = sd[f"net.{last}.bias"]
    return new_sd


def _infer_layer_configs(sd: dict) -> list:
    """
    Read hidden_dim for every linear block directly from weight shapes.
    Only works for pure-Linear architectures; returns [] for mixed types.
    """
    block_indices = sorted({
        int(m.group(1))
        for k in sd
        if (m := re.match(r"blocks\.(\d+)\.net\.0\.weight$", k))
    })
    return [
        {
            "type":       "linear",
            "hidden_dim": int(sd[f"blocks.{i}.net.0.weight"].shape[0]),
            "activation": "relu",
            "dropout":    0.3,
        }
        for i in block_indices
    ]


def _build_compatible_model(sd: dict, input_dim: int, saved_layer_configs: list):
    """
    Construct a FlexiblePPIModel whose architecture exactly matches `sd`.
    - Prefers shapes inferred from sd over saved_layer_configs (handles legacy).
    - Patches the output layer when the checkpoint used a 2-class head.
    Returns (model, num_output_classes).
    """
    # Prefer saved configs (complete architecture); fall back to weight-shape inference for legacy checkpoints only
    layer_configs = saved_layer_configs or _infer_layer_configs(sd)

    model = FlexiblePPIModel(input_dim, layer_configs)

    # Check output head size in the checkpoint
    saved_out_w = sd["output.weight"]        # shape: [num_classes, in_features]
    num_classes  = saved_out_w.shape[0]
    if num_classes != 1:
        # Old 2-class head — replace with matching linear
        model.output = nn.Linear(saved_out_w.shape[1], num_classes)

    return model, num_classes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

INFER_BATCH = 512   # rows per GPU forward pass


def _chunk_mask(chunks: torch.Tensor) -> torch.Tensor:
    return chunks.float().abs().sum(dim=-1).gt(0)


def run_inference(model_path: str, embedding_dict: dict, df: pd.DataFrame) -> list[dict]:
    """
    Score protein pairs using a saved FlexiblePPIModel checkpoint.

    Parameters
    ----------
    model_path      : path to .pt checkpoint saved by train_classifier
    embedding_dict  : {sequence -> torch.Tensor}
    df              : DataFrame with columns proteinA, proteinB

    Returns
    -------
    List of dicts: {proteinA, proteinB, probability, prediction, note}
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)

    input_dim  = ckpt.get("input_dim", 960)
    saved_cfgs = ckpt.get("layer_configs", [
        {"type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3},
        {"type": "linear", "hidden_dim": 64,  "activation": "relu", "dropout": 0.2},
    ])
    representation_mode = ckpt.get("embedding_representation", "pooled")

    # Prefer pair_mode saved in ckpt; otherwise infer from input_dim vs embedding size.
    sample_emb = next(iter(embedding_dict.values())) if embedding_dict else None
    actual_esm_dim = int(sample_emb.shape[-1]) if sample_emb is not None else (input_dim // 2)
    pair_mode = ckpt.get("pair_mode")
    if pair_mode not in ("concat", "product", "diff", "all"):
        if input_dim == 4 * actual_esm_dim:
            pair_mode = "all"
        elif input_dim == actual_esm_dim:
            pair_mode = "product"
        else:
            pair_mode = "concat"

    if representation_mode == "chunked":
        sd = ckpt["model_state"]
        model = FlexiblePPISequenceModel(input_dim, saved_cfgs)
        n_out = 1
    else:
        sd = _remap_legacy_state_dict(ckpt["model_state"])
        model, n_out = _build_compatible_model(sd, input_dim, saved_cfgs)
    model.load_state_dict(sd, strict=(representation_mode != "chunked"))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    results: list       = []
    batch_vecs: list    = []
    batch_masks: list   = []
    valid_indices: list = []

    for _, row in df.iterrows():
        seqA = str(row["proteinA"]).strip().upper()
        seqB = str(row["proteinB"]).strip().upper()
        eA   = embedding_dict.get(seqA)
        eB   = embedding_dict.get(seqB)

        if eA is None or eB is None:
            results.append({
                "proteinA":    seqA,
                "proteinB":    seqB,
                "probability": None,
                "prediction":  None,
                "note":        "embedding not available",
            })
            continue

        if representation_mode == "chunked":
            eA_f = eA.float()
            eB_f = eB.float()
            vec = torch.cat([eA_f, eB_f], dim=0)
            mask = torch.cat([_chunk_mask(eA_f), _chunk_mask(eB_f)], dim=0)
        else:
            eA_f = eA.float()
            eB_f = eB.float()
            mask = None
            if pair_mode == "all":
                vec = torch.cat([eA_f, eB_f, eA_f * eB_f, torch.abs(eA_f - eB_f)], dim=-1)
            elif pair_mode == "product":
                vec = eA_f * eB_f
            elif pair_mode == "diff":
                vec = torch.abs(eA_f - eB_f)
            else:
                vec = torch.cat([eA_f, eB_f], dim=-1)

        batch_vecs.append(vec)
        if mask is not None:
            batch_masks.append(mask)
        valid_indices.append(len(results))
        results.append({
            "proteinA":    seqA,
            "proteinB":    seqB,
            "probability": None,
            "prediction":  None,
            "note":        "",
        })

    # Batched GPU forward pass for all valid pairs
    if batch_vecs:
        all_probs: list = []
        with torch.no_grad():
            for i in range(0, len(batch_vecs), INFER_BATCH):
                batch  = torch.stack(batch_vecs[i : i + INFER_BATCH]).to(device)
                if representation_mode == "chunked":
                    masks = torch.stack(batch_masks[i : i + INFER_BATCH]).to(device)
                    logits = model(batch, masks)
                else:
                    logits = model(batch)
                if n_out == 1:
                    probs = torch.sigmoid(logits).squeeze(-1).cpu().tolist()
                else:
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                all_probs.extend(probs)

        for ri, prob in zip(valid_indices, all_probs):
            results[ri]["probability"] = round(prob, 4)
            results[ri]["prediction"]  = 1 if prob >= 0.5 else 0

    return results
