#########################################################################
'''
Author:        Dipayan <dipayansarkar26@gmail.com>
Licence:       MIT (see LICENCE file)
Description:   ESM2 embedding utilities (lazy-loaded, Celery-safe)
'''
#########################################################################

import os
import torch
import pandas as pd
import pickle
from esm import pretrained
from typing import List

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
# EMBED_LAYER = 33  # for esm2_t12_35M_UR650M
EMBED_LAYER = 12  # for esm2_t12_35M_UR50D


# ---------------------------------------------------------------------
# Lazy-loaded globals (IMPORTANT)
# ---------------------------------------------------------------------
_model = None
_alphabet = None
_batch_converter = None
_device = None

# ---------------------------------------------------------------------
# Model loader (called ONLY inside Celery)
# ---------------------------------------------------------------------
def get_esm():
    global _model, _alphabet, _batch_converter, _device

    if _model is None:
        _model, _alphabet = pretrained.load_model_and_alphabet(
            "esm2_t12_35M_UR50D"
        )

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(_device)
        _model.eval()

        _batch_converter = _alphabet.get_batch_converter()

    return _model, _batch_converter, _device

# ---------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------
def load_all_sequences(files, col_a="proteinA", col_b="proteinB"):
    """Extract unique sequences from one or more CSV files."""
    seq_set = set()
    for file in files:
        df = pd.read_csv(file)
        seq_set.update(df[col_a].astype(str).str.strip().str.upper())
        seq_set.update(df[col_b].astype(str).str.strip().str.upper())
    seq_set.discard("NAN")
    return sorted(seq_set)
#-----------------------------------------------------
def unload_esm():
    global _model, _alphabet, _batch_converter, _device

    _model = None
    _alphabet = None
    _batch_converter = None
    _device = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    import gc
    gc.collect()


#_ ----




# ---------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------
@torch.inference_mode()
def _embed_many(pairs: List[tuple]) -> torch.Tensor:
    model, batch_converter, device = get_esm()

    _, _, toks = batch_converter(pairs)
    toks = toks.to(device)

    out = model(toks, repr_layers=[EMBED_LAYER], return_contacts=False)
    reps = out["representations"][EMBED_LAYER]

    outs = []
    for i, (_, s) in enumerate(pairs):
        outs.append(reps[i, 1:len(s)+1].mean(dim=0))

    return torch.stack(outs)

def _embed_long_sliding(seq: str) -> torch.Tensor:
    chunks = [seq[i:i + MAX_ESM_LEN] for i in range(0, len(seq), STRIDE)]
    chunks[-1] = chunks[-1][:MAX_ESM_LEN]

    reps = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = [(f"w{i+j}", s) for j, s in enumerate(chunks[i:i+BATCH_SIZE])]
        reps.append(_embed_many(batch))

    return torch.cat(reps, dim=0).mean(dim=0).cpu()

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def compute_and_save_embeddings(all_sequences, outfile):
    try:
        embedding_dict = {}

        short_seqs = [s for s in all_sequences if len(s) <= MAX_ESM_LEN]
        long_seqs = [s for s in all_sequences if len(s) > MAX_ESM_LEN]

        for i in range(0, len(short_seqs), BATCH_SIZE):
            batch = [(f"seq{i+j}", s) for j, s in enumerate(short_seqs[i:i+BATCH_SIZE])]
            reps = _embed_many(batch)
            for (_, seq), rep in zip(batch, reps):
                embedding_dict[seq] = rep.cpu()

        for seq in long_seqs:
            embedding_dict[seq] = _embed_long_sliding(seq)

        with open(outfile, "wb") as f:
            pickle.dump(embedding_dict, f)

    finally:
        # ALWAYS unload model, even if errors happen
        unload_esm()
















