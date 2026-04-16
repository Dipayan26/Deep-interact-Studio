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
    # Prefer weight-shape inference; fall back to saved configs
    layer_configs = _infer_layer_configs(sd) or saved_layer_configs

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
    ckpt = torch.load(model_path, map_location="cpu")

    input_dim  = ckpt.get("input_dim", 960)
    saved_cfgs = ckpt.get("layer_configs", [
        {"type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3},
        {"type": "linear", "hidden_dim": 64,  "activation": "relu", "dropout": 0.2},
    ])

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

    sd           = _remap_legacy_state_dict(ckpt["model_state"])
    model, n_out = _build_compatible_model(sd, input_dim, saved_cfgs)
    model.load_state_dict(sd)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    results = []
    with torch.no_grad():
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

            eA_f = eA.float()
            eB_f = eB.float()
            if pair_mode == "all":
                vec = torch.cat([eA_f, eB_f, eA_f * eB_f, torch.abs(eA_f - eB_f)], dim=-1)
            elif pair_mode == "product":
                vec = eA_f * eB_f
            elif pair_mode == "diff":
                vec = torch.abs(eA_f - eB_f)
            else:
                vec = torch.cat([eA_f, eB_f], dim=-1)

            vec   = vec.unsqueeze(0).to(device)
            logit = model(vec)

            if n_out == 1:
                prob = torch.sigmoid(logit).item()
            else:
                # Legacy 2-class head: softmax probability of class 1
                prob = torch.softmax(logit, dim=-1)[0, 1].item()

            results.append({
                "proteinA":    seqA,
                "proteinB":    seqB,
                "probability": round(prob, 4),
                "prediction":  1 if prob >= 0.5 else 0,
                "note":        "",
            })

    return results
