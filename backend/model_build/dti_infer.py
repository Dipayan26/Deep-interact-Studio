"""
DTI inference: load a trained DTI checkpoint and score compound-protein pairs.
"""

import torch
import pandas as pd

from model_build.ppi_classifier import FlexiblePPIModel


def run_dti_inference(
    model_path: str,
    chem_dict: dict,
    esm_dict: dict,
    df: pd.DataFrame,
) -> list:
    """
    Score compound-protein pairs using a saved DTI checkpoint.

    Parameters
    ----------
    model_path : path to .pt checkpoint saved by train_dti_classifier
    chem_dict  : {smiles_str -> torch.Tensor}  (ChemBERTa embeddings)
    esm_dict   : {sequence_str -> torch.Tensor} (ESM2 embeddings)
    df         : DataFrame with columns 'smiles', 'sequence'

    Returns
    -------
    List of dicts: {smiles, sequence, probability, prediction, note}
    """
    ckpt = torch.load(model_path, map_location="cpu")
    input_dim    = int(ckpt.get("input_dim", 864))
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
            smiles = str(row["smiles"]).strip()
            seq    = str(row["sequence"]).strip().upper()
            e_chem = chem_dict.get(smiles)
            e_prot = esm_dict.get(seq)

            if e_chem is None or e_prot is None:
                missing = []
                if e_chem is None:
                    missing.append("compound embedding")
                if e_prot is None:
                    missing.append("protein embedding")
                results.append({
                    "smiles":      smiles,
                    "sequence":    seq[:40] + ("..." if len(seq) > 40 else ""),
                    "probability": None,
                    "prediction":  None,
                    "note":        f"missing: {', '.join(missing)}",
                })
                continue

            vec   = torch.cat([e_chem.float(), e_prot.float()], dim=-1).unsqueeze(0).to(device)
            logit = model(vec)
            prob  = torch.sigmoid(logit).item()

            results.append({
                "smiles":      smiles,
                "sequence":    seq[:40] + ("..." if len(seq) > 40 else ""),
                "probability": round(prob, 4),
                "prediction":  1 if prob >= 0.5 else 0,
                "note":        "",
            })

    return results
