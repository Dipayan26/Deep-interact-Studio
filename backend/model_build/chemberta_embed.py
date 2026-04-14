"""
ChemBERTa SMILES embedding utilities.
Model: seyonec/ChemBERTa-zinc-base-v1  (384-dim mean-pooled output)
Mirrors the lazy-load / unload pattern used in esm_embed.py.
"""

import gc
import os
import pickle
from typing import List

# Must be set BEFORE importing transformers/huggingface_hub — those libraries
# read HF_HOME at module-import time to compute the cache path constant.
os.environ.setdefault("HF_HOME", "/app/hf_cache")
os.makedirs("/app/hf_cache", exist_ok=True)

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

CHEMBERTA_DIM = 384          # seyonec/ChemBERTa-zinc-base-v1 hidden_size (6-layer, 384-dim)
CHEMBERTA_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Lazy-loaded globals (one model cached at a time)
# ---------------------------------------------------------------------------
_chem_model      = None
_chem_tokenizer  = None
_chem_device     = None
_loaded_chem_name = None


# ---------------------------------------------------------------------------
# Model loader / switcher
# ---------------------------------------------------------------------------
def get_chemberta(model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
    global _chem_model, _chem_tokenizer, _chem_device, _loaded_chem_name

    if _chem_model is not None and _loaded_chem_name != model_name:
        unload_chemberta()

    if _chem_model is None:
        print(f"[ChemBERTa] Loading {model_name} ...", flush=True)
        _chem_tokenizer  = AutoTokenizer.from_pretrained(model_name)
        _chem_model      = AutoModel.from_pretrained(model_name)
        _chem_device     = "cuda" if torch.cuda.is_available() else "cpu"
        _chem_model      = _chem_model.to(_chem_device)
        _chem_model.eval()
        _loaded_chem_name = model_name
        print(f"[ChemBERTa] Loaded on {_chem_device}", flush=True)

    return _chem_model, _chem_tokenizer, _chem_device


# ---------------------------------------------------------------------------
# Unloader
# ---------------------------------------------------------------------------
def unload_chemberta():
    global _chem_model, _chem_tokenizer, _chem_device, _loaded_chem_name
    _chem_model      = None
    _chem_tokenizer  = None
    _chem_device     = None
    _loaded_chem_name = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------
def load_all_smiles(files, col: str = "smiles") -> List[str]:
    """Extract unique SMILES strings from one or more CSV files."""
    smiles_set: set = set()
    for file in files:
        df = pd.read_csv(file)
        smiles_set.update(df[col].astype(str).str.strip())
    smiles_set.discard("nan")
    smiles_set.discard("")
    return sorted(smiles_set)


# ---------------------------------------------------------------------------
# Internal embedding helper
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _embed_smiles_batch(
    smiles_list: List[str],
    model,
    tokenizer,
    device: str,
) -> torch.Tensor:
    enc = tokenizer(
        smiles_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)

    # Mean pool over non-padding token embeddings
    token_embs = out.last_hidden_state                             # (B, T, D)
    attn_mask  = enc["attention_mask"].unsqueeze(-1).float()       # (B, T, 1)
    mean_emb   = (token_embs * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1e-9)
    return mean_emb.cpu()                                          # (B, D)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_and_save_chem_embeddings(
    all_smiles: List[str],
    outfile: str,
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
) -> None:
    """Compute ChemBERTa embeddings for all SMILES and save to a pickle file."""
    try:
        model, tokenizer, device = get_chemberta(model_name)
        embedding_dict: dict = {}

        for i in range(0, len(all_smiles), CHEMBERTA_BATCH_SIZE):
            batch = all_smiles[i : i + CHEMBERTA_BATCH_SIZE]
            reps  = _embed_smiles_batch(batch, model, tokenizer, device)
            for smi, rep in zip(batch, reps):
                embedding_dict[smi] = rep
            print(
                f"[ChemBERTa] Embedded {min(i + CHEMBERTA_BATCH_SIZE, len(all_smiles))}"
                f"/{len(all_smiles)} SMILES",
                flush=True,
            )

        with open(outfile, "wb") as f:
            pickle.dump(embedding_dict, f)
        print(f"[ChemBERTa] Embeddings saved → {outfile}", flush=True)

    finally:
        unload_chemberta()
