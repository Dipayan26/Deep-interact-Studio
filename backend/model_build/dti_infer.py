"""
DTI inference: load a trained DTI checkpoint and score compound-protein pairs.
"""

import torch
import pandas as pd

from model_build.ppi_classifier import FlexiblePPIModel

INFER_BATCH = 512   # rows per GPU forward pass


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
    input_dim    = int(ckpt.get("input_dim", 1248))  # ChemBERTa(768) + ESM2-35M(480)
    layer_configs = ckpt.get("layer_configs", [
        {"type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3},
        {"type": "linear", "hidden_dim": 64,  "activation": "relu", "dropout": 0.2},
    ])

    model = FlexiblePPIModel(input_dim, layer_configs)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    results: list = []
    batch_vecs: list = []
    valid_indices: list = []

    for _, row in df.iterrows():
        smiles = str(row["smiles"]).strip()
        seq    = str(row["sequence"]).strip().upper()
        e_chem = chem_dict.get(smiles)
        e_prot = esm_dict.get(seq)

        seq_display = seq[:40] + ("..." if len(seq) > 40 else "")

        if e_chem is None or e_prot is None:
            missing = []
            if e_chem is None:
                missing.append("compound embedding")
            if e_prot is None:
                missing.append("protein embedding")
            results.append({
                "smiles":      smiles,
                "sequence":    seq_display,
                "probability": None,
                "prediction":  None,
                "note":        f"missing: {', '.join(missing)}",
            })
        else:
            vec = torch.cat([e_chem.float(), e_prot.float()], dim=-1)
            batch_vecs.append(vec)
            valid_indices.append(len(results))
            results.append({
                "smiles":      smiles,
                "sequence":    seq_display,
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
                logits = model(batch)
                probs  = torch.sigmoid(logits).squeeze(-1).cpu().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                all_probs.extend(probs)

        for ri, prob in zip(valid_indices, all_probs):
            results[ri]["probability"] = round(prob, 4)
            results[ri]["prediction"]  = 1 if prob >= 0.5 else 0

    return results
