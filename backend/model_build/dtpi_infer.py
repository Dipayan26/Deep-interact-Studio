"""
DTPI inference: load a trained DTPI checkpoint and score compound-protein pairs.
"""

import torch
import pandas as pd

from model_build.inference_batching import (
    apply_pair_probabilities,
    collect_unique_valid_pairs,
    score_chunked_pairs,
    score_pooled_pairs,
)
from model_build.ppi_classifier import FlexiblePPIModel
from model_build.sequence_models import FlexiblePairSequenceModel

INFER_BATCH = 512   # rows per GPU forward pass


def run_dtpi_inference(
    model_path: str,
    chem_dict: dict,
    esm_dict: dict,
    df: pd.DataFrame,
) -> list:
    """
    Score compound-protein pairs using a saved DTPI checkpoint.

    Parameters
    ----------
    model_path : path to .pt checkpoint saved by train_dtpi_classifier
    chem_dict  : {smiles_str -> torch.Tensor}  (ChemBERTa embeddings)
    esm_dict   : {sequence_str -> torch.Tensor} (ESM2 embeddings)
    df         : DataFrame with columns 'smiles', 'sequence'

    Returns
    -------
    List of dicts: {smiles, sequence, probability, prediction, note}
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    input_dim    = int(ckpt.get("input_dim", 1248))  # ChemBERTa(768) + ESM2-35M(480)
    hyperparams = ckpt.get("hyperparams", {})
    representation_mode = str(
        ckpt.get("embedding_representation", hyperparams.get("embedding_representation", "pooled"))
    ).lower()
    layer_configs = ckpt.get("layer_configs", [
        {"type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3},
        {"type": "linear", "hidden_dim": 64,  "activation": "relu", "dropout": 0.2},
    ])

    if representation_mode == "chunked":
        model = FlexiblePairSequenceModel(
            int(ckpt.get("chem_dim", hyperparams.get("chem_dim", 768))),
            int(ckpt.get("esm_dim", hyperparams.get("esm_dim", 480))),
            int(ckpt.get("chunk_model_dim", input_dim)),
            layer_configs,
        )
    else:
        model = FlexiblePPIModel(input_dim, layer_configs)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    smiles_values = df["smiles"].astype(str).str.strip().tolist()
    seq_values = df["sequence"].astype(str).str.strip().str.upper().tolist()
    pairs_by_row = list(zip(smiles_values, seq_values))
    unique_pairs, _, availability = collect_unique_valid_pairs(
        smiles_values, seq_values, chem_dict, esm_dict
    )
    results: list = []
    for (smiles, seq), (has_chem, has_prot) in zip(pairs_by_row, availability):
        missing = []
        if not has_chem:
            missing.append("compound embedding")
        if not has_prot:
            missing.append("protein embedding")
        results.append({
            "smiles": smiles,
            "sequence": seq[:40] + ("..." if len(seq) > 40 else ""),
            "probability": None,
            "prediction": None,
            "note": "" if not missing else f"missing: {', '.join(missing)}",
        })

    if unique_pairs:
        if representation_mode == "chunked":
            pair_probs = score_chunked_pairs(
                unique_pairs, chem_dict, esm_dict, model, device, INFER_BATCH
            )
        else:
            pair_probs = score_pooled_pairs(
                unique_pairs,
                chem_dict,
                esm_dict,
                model,
                device,
                INFER_BATCH,
                lambda left, right: torch.cat([left, right], dim=-1),
            )
        apply_pair_probabilities(results, pairs_by_row, pair_probs)

    return results
