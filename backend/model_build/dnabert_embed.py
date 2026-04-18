"""
DNABERT embedding utilities.
Model: armheb/DNA_bert_6 (768-dim mean-pooled output)
Standard BERT with 6-mer tokenization — no custom code, compatible with transformers 5.x.
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
from transformers import AutoTokenizer, AutoModel

DNABERT_DIM        = 768
DNABERT_BATCH_SIZE = 16
DNABERT_DEFAULT    = "armheb/DNA_bert_6"

# ---------------------------------------------------------------------------
# Lazy-loaded globals (one model cached at a time)
# ---------------------------------------------------------------------------
_dna_model     = None
_dna_tokenizer = None
_dna_device    = None
_loaded_dna_name: str | None = None


# ---------------------------------------------------------------------------
# 6-mer tokenization helper required by DNABERT
# ---------------------------------------------------------------------------
def _seq_to_kmers(seq: str, k: int = 6) -> str:
    """Split a DNA sequence into space-separated k-mers (stride 1)."""
    return " ".join(seq[i : i + k] for i in range(len(seq) - k + 1)) if len(seq) >= k else seq


# ---------------------------------------------------------------------------
# Model loader / switcher
# ---------------------------------------------------------------------------
def get_dnabert(model_name: str = DNABERT_DEFAULT):
    global _dna_model, _dna_tokenizer, _dna_device, _loaded_dna_name

    if _dna_model is not None and _loaded_dna_name != model_name:
        unload_dnabert()

    if _dna_model is None:
        print(f"[DNABERT] Loading {model_name} ...", flush=True)
        _dna_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _dna_model     = AutoModel.from_pretrained(model_name)
        _dna_device    = "cuda" if torch.cuda.is_available() else "cpu"
        _dna_model     = _dna_model.to(_dna_device)
        _dna_model.eval()
        _loaded_dna_name = model_name
        print(f"[DNABERT] Loaded on {_dna_device}", flush=True)

    assert _dna_device is not None
    return _dna_model, _dna_tokenizer, _dna_device


# ---------------------------------------------------------------------------
# Unloader
# ---------------------------------------------------------------------------
def unload_dnabert():
    global _dna_model, _dna_tokenizer, _dna_device, _loaded_dna_name
    _dna_model       = None
    _dna_tokenizer   = None
    _dna_device      = None
    _loaded_dna_name = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------
def load_all_dna_sequences(files, col: str = "dna_sequence") -> List[str]:
    """Extract unique DNA sequences from one or more CSV files."""
    dna_set: set = set()
    for file in files:
        df = pd.read_csv(file)
        dna_set.update(
            df[col].astype(str).str.strip().str.upper()
        )
    dna_set.discard("NAN")
    dna_set.discard("")
    return sorted(dna_set)


# ---------------------------------------------------------------------------
# Internal embedding helper
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _embed_dna_batch(
    dna_list: List[str],
    model,
    tokenizer,
    device: str,
) -> torch.Tensor:
    # DNABERT tokenizer expects space-separated 6-mers
    kmer_seqs = [_seq_to_kmers(s) for s in dna_list]
    enc = tokenizer(
        kmer_seqs,
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
def compute_and_save_dna_embeddings(
    all_dna: List[str],
    outfile: str,
    model_name: str = DNABERT_DEFAULT,
) -> None:
    """Compute DNABERT embeddings for all DNA sequences and save to a pickle file."""
    try:
        model, tokenizer, device = get_dnabert(model_name)
        embedding_dict: dict = {}

        for i in range(0, len(all_dna), DNABERT_BATCH_SIZE):
            batch = all_dna[i : i + DNABERT_BATCH_SIZE]
            reps  = _embed_dna_batch(batch, model, tokenizer, device)
            for dna, rep in zip(batch, reps):
                embedding_dict[dna] = rep
            print(
                f"[DNABERT] Embedded {min(i + DNABERT_BATCH_SIZE, len(all_dna))}"
                f"/{len(all_dna)} DNA sequences",
                flush=True,
            )

        with open(outfile, "wb") as f:
            pickle.dump(embedding_dict, f)
        print(f"[DNABERT] Embeddings saved → {outfile}", flush=True)

    finally:
        unload_dnabert()
