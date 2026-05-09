"""
RPI inference: load a trained RPI checkpoint and score RNA-protein pairs.
"""

import torch
import pandas as pd

from model_build.chunked_pair_classifier import chunk_mask
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

    results: list      = []
    batch_vecs: list   = []
    batch_left: list = []
    batch_right: list = []
    batch_left_masks: list = []
    batch_right_masks: list = []
    valid_indices: list = []

    for _, row in df.iterrows():
        rna_seq  = str(row["rna_sequence"]).strip().upper().replace("T", "U")
        prot_seq = str(row["protein_sequence"]).strip().upper()
        e_rna    = rna_dict.get(rna_seq)
        e_prot   = esm_dict.get(prot_seq)

        rna_display  = rna_seq[:40]  + ("..." if len(rna_seq)  > 40 else "")
        prot_display = prot_seq[:40] + ("..." if len(prot_seq) > 40 else "")

        if e_rna is None or e_prot is None:
            missing = []
            if e_rna  is None:
                missing.append("RNA embedding")
            if e_prot is None:
                missing.append("protein embedding")
            results.append({
                "rna_sequence":     rna_display,
                "protein_sequence": prot_display,
                "probability":      None,
                "prediction":       None,
                "note":             f"missing: {', '.join(missing)}",
            })
        else:
            if representation_mode == "chunked":
                left = e_rna.float()
                right = e_prot.float()
                batch_left.append(left)
                batch_right.append(right)
                batch_left_masks.append(chunk_mask(left))
                batch_right_masks.append(chunk_mask(right))
            else:
                vec = torch.cat([e_rna.float(), e_prot.float()], dim=-1)
                batch_vecs.append(vec)
            valid_indices.append(len(results))
            results.append({
                "rna_sequence":     rna_display,
                "protein_sequence": prot_display,
                "probability":      None,
                "prediction":       None,
                "note":             "",
            })

    # Batched GPU forward pass for all valid pairs
    if representation_mode == "chunked" and batch_left:
        all_probs: list = []
        with torch.no_grad():
            for i in range(0, len(batch_left), INFER_BATCH):
                left = torch.stack(batch_left[i : i + INFER_BATCH]).to(device)
                right = torch.stack(batch_right[i : i + INFER_BATCH]).to(device)
                left_mask = torch.stack(batch_left_masks[i : i + INFER_BATCH]).to(device)
                right_mask = torch.stack(batch_right_masks[i : i + INFER_BATCH]).to(device)
                logits = model(left, right, left_mask, right_mask)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                all_probs.extend(probs)

        for ri, prob in zip(valid_indices, all_probs):
            results[ri]["probability"] = round(prob, 4)
            results[ri]["prediction"] = 1 if prob >= 0.5 else 0

    elif batch_vecs:
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
