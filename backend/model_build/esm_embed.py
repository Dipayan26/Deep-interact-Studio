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


@torch.inference_mode()
def _embed_chunked_batch(
    pairs: List[tuple],
    model,
    batch_converter,
    device: str,
    embed_layer: int,
    max_len: int,
    num_chunks: int,
) -> dict:
    _, _, toks = batch_converter(pairs)
    toks = toks.to(device)

    out = model(toks, repr_layers=[embed_layer], return_contacts=False)
    reps = out["representations"][embed_layer]

    chunk_size = max(1, int((max_len + num_chunks - 1) // num_chunks))
    chunked = {}
    for i, (_, seq) in enumerate(pairs):
        token_reps = reps[i, 1 : min(len(seq), max_len) + 1]
        chunks = []
        for chunk_i in range(num_chunks):
            start = chunk_i * chunk_size
            end = min(start + chunk_size, token_reps.shape[0])
            if start < token_reps.shape[0] and end > start:
                chunks.append(token_reps[start:end].mean(dim=0))
            else:
                chunks.append(torch.zeros(token_reps.shape[-1], device=device, dtype=token_reps.dtype))
        chunked[seq] = torch.stack(chunks, dim=0).cpu()
    return chunked


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


def _embed_long_chunked(
    seq: str,
    model,
    batch_converter,
    device: str,
    embed_layer: int,
    max_len: int,
    num_chunks: int,
) -> torch.Tensor:
    capped = seq[:max_len]
    chunk_size = max(1, int((max_len + num_chunks - 1) // num_chunks))
    chunks = []
    for chunk_i in range(num_chunks):
        start = chunk_i * chunk_size
        end = min(start + chunk_size, len(capped))
        subseq = capped[start:end]
        if not subseq:
            dim = ESM2_MODELS[_loaded_model_name or "esm2_t12_35M_UR50D"]["dim"]
            chunks.append(torch.zeros(dim))
            continue
        batch = [(f"chunk{chunk_i}", subseq)]
        reps = _embed_batch(batch, model, batch_converter, device, embed_layer)
        chunks.append(reps[0].cpu())
    return torch.stack(chunks, dim=0)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def compute_and_save_embeddings(
    all_sequences: List[str],
    outfile: str,
    model_name: str = "esm2_t12_35M_UR50D",
    progress_callback=None,
) -> None:
    """Compute ESM2 embeddings for all sequences and save to a pickle file."""
    try:
        model, batch_converter, device, embed_layer = get_esm(model_name)
        embedding_dict = {}

        short_seqs = [s for s in all_sequences if len(s) <= MAX_ESM_LEN]
        long_seqs  = [s for s in all_sequences if len(s) > MAX_ESM_LEN]
        total = len(short_seqs) + len(long_seqs)
        done = 0

        for i in range(0, len(short_seqs), BATCH_SIZE):
            batch = [(f"seq{i + j}", s) for j, s in enumerate(short_seqs[i:i + BATCH_SIZE])]
            reps = _embed_batch(batch, model, batch_converter, device, embed_layer)
            for (_, seq), rep in zip(batch, reps):
                embedding_dict[seq] = rep.cpu()
            done += len(batch)
            if progress_callback is not None:
                progress_callback(done, total, "Embedding ESM2 protein sequences")

        for seq in long_seqs:
            embedding_dict[seq] = _embed_long(seq, model, batch_converter, device, embed_layer)
            done += 1
            if progress_callback is not None:
                progress_callback(done, total, "Embedding long ESM2 protein sequences")

        with open(outfile, "wb") as f:
            pickle.dump(embedding_dict, f)

    finally:
        # ALWAYS unload model, even if errors happen
        unload_esm()


def compute_and_save_chunked_embeddings(
    all_sequences: List[str],
    outfile: str,
    model_name: str = "esm2_t12_35M_UR50D",
    max_len: int = 512,
    num_chunks: int = 8,
    dtype: str = "float16",
    progress_callback=None,
) -> None:
    """Compute fixed-size chunk-pooled ESM2 embeddings and save to pickle.

    Each sequence maps to a tensor of shape (num_chunks, esm_dim). Chunks are
    local mean pools over the first max_len residues.
    """
    max_len = max(1, min(int(max_len), MAX_ESM_LEN))
    num_chunks = max(1, int(num_chunks))
    use_fp16 = str(dtype).lower() in {"fp16", "float16", "half"}

    try:
        model, batch_converter, device, embed_layer = get_esm(model_name)
        embedding_dict = {}

        short_seqs = [s for s in all_sequences if len(s) <= max_len]
        long_seqs = [s for s in all_sequences if len(s) > max_len]
        total = len(short_seqs) + len(long_seqs)
        done = 0

        for i in range(0, len(short_seqs), BATCH_SIZE):
            batch = [(f"seq{i + j}", s) for j, s in enumerate(short_seqs[i:i + BATCH_SIZE])]
            chunked = _embed_chunked_batch(
                batch, model, batch_converter, device, embed_layer, max_len, num_chunks
            )
            for seq, rep in chunked.items():
                embedding_dict[seq] = rep.half() if use_fp16 else rep.float()
            done += len(batch)
            print(
                f"[ESM2 chunked] Embedded {min(i + BATCH_SIZE, len(short_seqs))}"
                f"/{len(short_seqs)} short sequences",
                flush=True,
            )
            if progress_callback is not None:
                progress_callback(done, total, "Embedding chunked ESM2 protein sequences")

        for i, seq in enumerate(long_seqs, start=1):
            rep = _embed_long_chunked(seq, model, batch_converter, device, embed_layer, max_len, num_chunks)
            embedding_dict[seq] = rep.half() if use_fp16 else rep.float()
            done += 1
            print(f"[ESM2 chunked] Embedded long {i}/{len(long_seqs)}", flush=True)
            if progress_callback is not None:
                progress_callback(done, total, "Embedding long chunked ESM2 protein sequences")

        with open(outfile, "wb") as f:
            pickle.dump(embedding_dict, f)

    finally:
        unload_esm()
