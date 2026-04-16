"""
RNA-FM embedding utilities.
Model: multimolecule/rnafm  (640-dim mean-pooled output)
Mirrors the lazy-load / unload pattern used in chemberta_embed.py.
"""

import gc
import os
import pickle
from typing import List

# Must be set BEFORE importing transformers/huggingface_hub
os.environ.setdefault("HF_HOME", "/app/hf_cache")
os.makedirs("/app/hf_cache", exist_ok=True)

import torch
import pandas as pd
from multimolecule import RnaFmModel, RnaTokenizer

RNAFM_DIM        = 640
RNAFM_BATCH_SIZE = 16
RNAFM_DEFAULT    = "multimolecule/rnafm"

# ---------------------------------------------------------------------------
# Lazy-loaded globals (one model cached at a time)
# ---------------------------------------------------------------------------
_rna_model     = None
_rna_tokenizer = None
_rna_device    = None
_loaded_rna_name: str | None = None


# ---------------------------------------------------------------------------
# Model loader / switcher
# ---------------------------------------------------------------------------
def get_rnafm(model_name: str = RNAFM_DEFAULT):
    global _rna_model, _rna_tokenizer, _rna_device, _loaded_rna_name

    if _rna_model is not None and _loaded_rna_name != model_name:
        unload_rnafm()

    if _rna_model is None:
        print(f"[RNA-FM] Loading {model_name} ...", flush=True)
        _rna_tokenizer = RnaTokenizer.from_pretrained(model_name)
        _rna_model     = RnaFmModel.from_pretrained(model_name)
        _rna_device    = "cuda" if torch.cuda.is_available() else "cpu"
        _rna_model     = _rna_model.to(_rna_device)
        _rna_model.eval()
        _loaded_rna_name = model_name
        print(f"[RNA-FM] Loaded on {_rna_device}", flush=True)

    return _rna_model, _rna_tokenizer, _rna_device


# ---------------------------------------------------------------------------
# Unloader
# ---------------------------------------------------------------------------
def unload_rnafm():
    global _rna_model, _rna_tokenizer, _rna_device, _loaded_rna_name
    _rna_model       = None
    _rna_tokenizer   = None
    _rna_device      = None
    _loaded_rna_name = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------
def load_all_rna_sequences(files, col: str = "rna_sequence") -> List[str]:
    """Extract unique RNA sequences from one or more CSV files."""
    rna_set: set = set()
    for file in files:
        df = pd.read_csv(file)
        rna_set.update(
            df[col].astype(str).str.strip().str.upper().str.replace("T", "U", regex=False)
        )
    rna_set.discard("NAN")
    rna_set.discard("")
    return sorted(rna_set)


# ---------------------------------------------------------------------------
# Internal embedding helper
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _embed_rna_batch(
    rna_list: List[str],
    model,
    tokenizer,
    device: str,
) -> torch.Tensor:
    # RNA-FM tokenizer expects uppercase A/U/G/C
    enc = tokenizer(
        rna_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)

    # Mean pool over non-padding token embeddings
    token_embs = out.last_hidden_state                              # (B, T, D)
    attn_mask  = enc["attention_mask"].unsqueeze(-1).float()        # (B, T, 1)
    mean_emb   = (token_embs * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1e-9)
    return mean_emb.cpu()                                           # (B, D)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_and_save_rna_embeddings(
    all_rna: List[str],
    outfile: str,
    model_name: str = RNAFM_DEFAULT,
) -> None:
    """Compute RNA-FM embeddings for all RNA sequences and save to a pickle file."""
    try:
        model, tokenizer, device = get_rnafm(model_name)
        embedding_dict: dict = {}

        for i in range(0, len(all_rna), RNAFM_BATCH_SIZE):
            batch = all_rna[i : i + RNAFM_BATCH_SIZE]
            reps  = _embed_rna_batch(batch, model, tokenizer, device)
            for rna, rep in zip(batch, reps):
                embedding_dict[rna] = rep
            print(
                f"[RNA-FM] Embedded {min(i + RNAFM_BATCH_SIZE, len(all_rna))}"
                f"/{len(all_rna)} RNA sequences",
                flush=True,
            )

        with open(outfile, "wb") as f:
            pickle.dump(embedding_dict, f)
        print(f"[RNA-FM] Embeddings saved → {outfile}", flush=True)

    finally:
        unload_rnafm()
