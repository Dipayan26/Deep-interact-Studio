"""
Inference utilities: load a trained PPIClassifier and score new pairs.
"""

import torch
import pandas as pd
from model_build.ppi_classifier import PPIClassifier, make_pair_vector


def run_inference(model_path: str, embedding_dict: dict, df: pd.DataFrame) -> list[dict]:
    """
    Score protein pairs using a saved PPIClassifier checkpoint.

    Parameters
    ----------
    model_path      : path to .pt checkpoint saved by train_classifier
    embedding_dict  : {sequence -> torch.Tensor}  (may include newly generated ones)
    df              : DataFrame with columns proteinA, proteinB

    Returns
    -------
    List of dicts: {proteinA, proteinB, probability, prediction}
    """

    ckpt = torch.load(model_path, map_location="cpu")
    hp        = ckpt["hyperparams"]
    method    = hp.get("pair_method", "all")
    hidden_dim = int(ckpt.get("hidden_dim", hp.get("hidden_dim", 256)))
    num_layers = int(ckpt.get("num_layers", hp.get("num_layers", 2)))
    input_dim  = ckpt["input_dim"]

    model = PPIClassifier(input_dim, hidden_dim, num_layers)
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

            vec  = make_pair_vector(eA, eB, method).float().unsqueeze(0).to(device)
            out  = model(vec)
            prob = torch.softmax(out, dim=1)[0, 1].item()
            results.append({
                "proteinA":    seqA,
                "proteinB":    seqB,
                "probability": round(prob, 4),
                "prediction":  1 if prob >= 0.5 else 0,
                "note":        "",
            })

    return results
