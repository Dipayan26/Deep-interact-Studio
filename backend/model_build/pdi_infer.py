"""
PDI inference: load a trained PDI checkpoint and score protein-DNA pairs.
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


def run_pdi_inference(
    model_path: str,
    dna_dict: dict,
    esm_dict: dict,
    df: pd.DataFrame,
) -> list:
    """
    Score protein-DNA pairs using a saved PDI checkpoint.

    Parameters
    ----------
    model_path : path to .pt checkpoint saved by train_pdi_classifier
    dna_dict   : {dna_seq_str -> torch.Tensor}      (DNABERT-2 embeddings)
    esm_dict   : {protein_seq_str -> torch.Tensor}  (ESM2 embeddings)
    df         : DataFrame with columns 'dna_sequence', 'protein_sequence'

    Returns
    -------
    List of dicts: {dna_sequence, protein_sequence, probability, prediction, note}
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    input_dim     = int(ckpt.get("input_dim", 1248))   # DNABERT-2(768) + ESM2-35M(480)
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
            int(ckpt.get("dna_dim", hyperparams.get("dna_dim", 768))),
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

    dna_values = df["dna_sequence"].astype(str).str.strip().str.upper().tolist()
    prot_values = df["protein_sequence"].astype(str).str.strip().str.upper().tolist()
    pairs_by_row = list(zip(dna_values, prot_values))
    unique_pairs, _, availability = collect_unique_valid_pairs(
        dna_values, prot_values, dna_dict, esm_dict
    )
    results: list = []
    for (dna_seq, prot_seq), (has_dna, has_prot) in zip(pairs_by_row, availability):
        missing = []
        if not has_dna:
            missing.append("DNA embedding")
        if not has_prot:
            missing.append("protein embedding")
        results.append({
            "dna_sequence": dna_seq[:40] + ("..." if len(dna_seq) > 40 else ""),
            "protein_sequence": prot_seq[:40] + ("..." if len(prot_seq) > 40 else ""),
            "probability": None,
            "prediction": None,
            "note": "" if not missing else f"missing: {', '.join(missing)}",
        })

    if unique_pairs:
        if representation_mode == "chunked":
            pair_probs = score_chunked_pairs(
                unique_pairs, dna_dict, esm_dict, model, device, INFER_BATCH
            )
        else:
            pair_probs = score_pooled_pairs(
                unique_pairs,
                dna_dict,
                esm_dict,
                model,
                device,
                INFER_BATCH,
                lambda left, right: torch.cat([left, right], dim=-1),
            )
        apply_pair_probabilities(results, pairs_by_row, pair_probs)

    return results
