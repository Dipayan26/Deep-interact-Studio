"""
Inference utilities: load a trained FlexiblePPIModel and score new pairs.
"""

import torch
import pandas as pd
from model_build.ppi_classifier import FlexiblePPIModel


def run_inference(model_path: str, embedding_dict: dict, df: pd.DataFrame) -> list[dict]:
    """
    Score protein pairs using a saved FlexiblePPIModel checkpoint.

    Parameters
    ----------
    model_path      : path to .pt checkpoint saved by train_classifier
    embedding_dict  : {sequence -> torch.Tensor}  (may include newly generated ones)
    df              : DataFrame with columns proteinA, proteinB

    Returns
    -------
    List of dicts: {proteinA, proteinB, probability, prediction, note}
    """

    ckpt         = torch.load(model_path, map_location="cpu")
    input_dim    = ckpt["input_dim"]
    layer_configs = ckpt.get("layer_configs", [
        {"type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3},
        {"type": "linear", "hidden_dim": 64,  "activation": "relu", "dropout": 0.2},
    ])

    model = FlexiblePPIModel(input_dim, layer_configs)
    model.load_state_dict(ckpt["model_state"])
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

            vec    = torch.cat([eA, eB], dim=-1).float().unsqueeze(0).to(device)
            logit  = model(vec)
            prob   = torch.sigmoid(logit).item()
            results.append({
                "proteinA":    seqA,
                "proteinB":    seqB,
                "probability": round(prob, 4),
                "prediction":  1 if prob >= 0.5 else 0,
                "note":        "",
            })

    return results
