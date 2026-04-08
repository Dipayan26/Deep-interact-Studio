#########################################################################
'''
Author:        Dipayan <dipayansarkar26@gmail.com>
Licence:       MIT (see LICENCE file)
Description:   ESM2 embedding utilities — multi-model, lazy-loaded, Celery-safe
'''
#########################################################################

import gc
import os
import pickle
from typing import List

import torch
import pandas as pd
from esm import pretrained

# ---------------------------------------------------------------------
# Torch cache (FIXES PermissionError: '/.cache')
# ---------------------------------------------------------------------
os.environ["TORCH_HOME"] = "/tmp/.cache/torch"
os.makedirs("/tmp/.cache/torch", exist_ok=True)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
MAX_ESM_LEN = 1022
STRIDE = 512
BATCH_SIZE = 8

ESM2_MODELS = {
    "esm2_t6_8M_UR50D":   {"dim": 320,  "layer": 6},
    "esm2_t12_35M_UR50D": {"dim": 480,  "layer": 12},
    "esm2_t30_150M_UR50D":{"dim": 640,  "layer": 30},
    "esm2_t33_650M_UR50D":{"dim": 1280, "layer": 33},
}

# ---------------------------------------------------------------------
# Lazy-loaded globals (one model cached at a time)
# ---------------------------------------------------------------------
_model = None
_batch_converter = None
_device = None
_embed_layer = None
_loaded_model_name = None


# ---------------------------------------------------------------------
# Model loader / switcher
# ---------------------------------------------------------------------
def get_esm(model_name: str = "esm2_t12_35M_UR50D"):
    """Load (or return cached) ESM2 model. Switches model if name differs."""
    global _model, _batch_converter, _device, _embed_layer, _loaded_model_name

    if model_name not in ESM2_MODELS:
        raise ValueError(
            f"Unknown ESM2 model '{model_name}'. "
            f"Choose from: {list(ESM2_MODELS.keys())}"
        )

    if _model is not None and _loaded_model_name != model_name:
        # Different model requested — unload current first
        unload_esm()

    if _model is None:
        cfg = ESM2_MODELS[model_name]
        _embed_layer = cfg["layer"]

        _model, alphabet = pretrained.load_model_and_alphabet(model_name)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(_device)
        _model.eval()
        _batch_converter = alphabet.get_batch_converter()
        _loaded_model_name = model_name

    return _model, _batch_converter, _device, _embed_layer


# ---------------------------------------------------------------------
# Unloader
# ---------------------------------------------------------------------
def unload_esm():
    global _model, _batch_converter, _device, _embed_layer, _loaded_model_name

    _model = None
    _batch_converter = None
    _device = None
    _embed_layer = None
    _loaded_model_name = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


# ---------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------
def load_all_sequences(files, col_a: str = "proteinA", col_b: str = "proteinB") -> List[str]:
    """Extract unique sequences from one or more CSV files."""
    seq_set = set()
    for file in files:
        df = pd.read_csv(file)
        seq_set.update(df[col_a].astype(str).str.strip().str.upper())
        seq_set.update(df[col_b].astype(str).str.strip().str.upper())
    seq_set.discard("NAN")
    return sorted(seq_set)


# ---------------------------------------------------------------------
# Internal embedding helpers (explicit params, no globals except cache)
# ---------------------------------------------------------------------
@torch.inference_mode()
def _embed_batch(
    pairs: List[tuple],
    model,
    batch_converter,
    device: str,
    embed_layer: int,
) -> torch.Tensor:
    _, _, toks = batch_converter(pairs)
    toks = toks.to(device)

    out = model(toks, repr_layers=[embed_layer], return_contacts=False)
    reps = out["representations"][embed_layer]

    outs = []
    for i, (_, s) in enumerate(pairs):
        outs.append(reps[i, 1:len(s) + 1].mean(dim=0))

    return torch.stack(outs)


def _embed_long(
    seq: str,
    model,
    batch_converter,
    device: str,
    embed_layer: int,
) -> torch.Tensor:
    chunks = [seq[i:i + MAX_ESM_LEN] for i in range(0, len(seq), STRIDE)]
    chunks[-1] = chunks[-1][:MAX_ESM_LEN]

    reps = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = [(f"w{i + j}", s) for j, s in enumerate(chunks[i:i + BATCH_SIZE])]
        reps.append(_embed_batch(batch, model, batch_converter, device, embed_layer))

    return torch.cat(reps, dim=0).mean(dim=0).cpu()


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def compute_and_save_embeddings(
    all_sequences: List[str],
    outfile: str,
    model_name: str = "esm2_t12_35M_UR50D",
) -> None:
    """Compute ESM2 embeddings for all sequences and save to a pickle file."""
    try:
        model, batch_converter, device, embed_layer = get_esm(model_name)
        embedding_dict = {}

        short_seqs = [s for s in all_sequences if len(s) <= MAX_ESM_LEN]
        long_seqs  = [s for s in all_sequences if len(s) > MAX_ESM_LEN]

        for i in range(0, len(short_seqs), BATCH_SIZE):
            batch = [(f"seq{i + j}", s) for j, s in enumerate(short_seqs[i:i + BATCH_SIZE])]
            reps = _embed_batch(batch, model, batch_converter, device, embed_layer)
            for (_, seq), rep in zip(batch, reps):
                embedding_dict[seq] = rep.cpu()

        for seq in long_seqs:
            embedding_dict[seq] = _embed_long(seq, model, batch_converter, device, embed_layer)

        with open(outfile, "wb") as f:
            pickle.dump(embedding_dict, f)

    finally:
        # ALWAYS unload model, even if errors happen
        unload_esm()
