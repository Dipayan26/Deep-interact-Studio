"""
RPI inference: load a trained RPI checkpoint and score RNA-protein pairs.
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


def run_rpi_inference(
    model_path: str,
    rna_dict: dict,
    esm_dict: dict,
    df: pd.DataFrame,
) -> list:
    """
    Score RNA-protein pairs using a saved RPI checkpoint.

    Parameters
    ----------
    model_path : path to .pt checkpoint saved by train_rpi_classifier
    rna_dict   : {rna_seq_str -> torch.Tensor}   (RNA-FM embeddings)
    esm_dict   : {protein_seq_str -> torch.Tensor} (ESM2 embeddings)
    df         : DataFrame with columns 'rna_sequence', 'protein_sequence'

    Returns
    -------
    List of dicts: {rna_sequence, protein_sequence, probability, prediction, note}
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    input_dim     = int(ckpt.get("input_dim", 1120))   # RNA-FM(640) + ESM2-35M(480)
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
            int(ckpt.get("rna_dim", hyperparams.get("rna_dim", 640))),
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

    rna_values = (
        df["rna_sequence"].astype(str).str.strip().str.upper()
        .str.replace("T", "U", regex=False).tolist()
    )
    prot_values = df["protein_sequence"].astype(str).str.strip().str.upper().tolist()
    pairs_by_row = list(zip(rna_values, prot_values))
    unique_pairs, _, availability = collect_unique_valid_pairs(
        rna_values, prot_values, rna_dict, esm_dict
    )
    results: list = []
    for (rna_seq, prot_seq), (has_rna, has_prot) in zip(pairs_by_row, availability):
        missing = []
        if not has_rna:
            missing.append("RNA embedding")
        if not has_prot:
            missing.append("protein embedding")
        results.append({
            "rna_sequence": rna_seq[:40] + ("..." if len(rna_seq) > 40 else ""),
            "protein_sequence": prot_seq[:40] + ("..." if len(prot_seq) > 40 else ""),
            "probability": None,
            "prediction": None,
            "note": "" if not missing else f"missing: {', '.join(missing)}",
        })

    if unique_pairs:
        if representation_mode == "chunked":
            pair_probs = score_chunked_pairs(
                unique_pairs, rna_dict, esm_dict, model, device, INFER_BATCH
            )
        else:
            pair_probs = score_pooled_pairs(
                unique_pairs,
                rna_dict,
                esm_dict,
                model,
                device,
                INFER_BATCH,
                lambda left, right: torch.cat([left, right], dim=-1),
            )
        apply_pair_probabilities(results, pairs_by_row, pair_probs)

    return results
