"""Shared batching helpers for interaction inference."""

from collections import OrderedDict
from typing import Callable, Iterable

import torch


Pair = tuple[str, str]


def collect_unique_valid_pairs(
    left_values: Iterable[str],
    right_values: Iterable[str],
    left_dict: dict,
    right_dict: dict,
) -> tuple[list[Pair], dict[Pair, list[int]], list[tuple[bool, bool]]]:
    """Return unique valid pair keys and per-row embedding availability."""
    unique_pairs: OrderedDict[Pair, None] = OrderedDict()
    pair_to_indices: dict[Pair, list[int]] = {}
    availability: list[tuple[bool, bool]] = []

    for idx, (left, right) in enumerate(zip(left_values, right_values)):
        has_left = left in left_dict
        has_right = right in right_dict
        availability.append((has_left, has_right))
        if not (has_left and has_right):
            continue
        pair = (left, right)
        unique_pairs.setdefault(pair, None)
        pair_to_indices.setdefault(pair, []).append(idx)

    return list(unique_pairs.keys()), pair_to_indices, availability


def chunk_mask(chunks: torch.Tensor) -> torch.Tensor:
    return chunks.float().abs().sum(dim=-1).gt(0)


def _as_prob_list(logits: torch.Tensor, n_out: int = 1) -> list[float]:
    if n_out == 1:
        probs = torch.sigmoid(logits).squeeze(-1).detach().cpu().tolist()
    else:
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
    if isinstance(probs, float):
        return [probs]
    return probs


def score_pooled_pairs(
    unique_pairs: list[Pair],
    left_dict: dict,
    right_dict: dict,
    model,
    device: str,
    batch_size: int,
    combine_batch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n_out: int = 1,
) -> dict[Pair, float]:
    """Score unique pooled-embedding pairs in GPU batches."""
    pair_probs: dict[Pair, float] = {}
    with torch.inference_mode():
        for start in range(0, len(unique_pairs), batch_size):
            pairs = unique_pairs[start : start + batch_size]
            left = torch.stack([left_dict[left].float() for left, _ in pairs]).to(device)
            right = torch.stack([right_dict[right].float() for _, right in pairs]).to(device)
            batch = combine_batch(left, right)
            probs = _as_prob_list(model(batch), n_out=n_out)
            for pair, prob in zip(pairs, probs):
                pair_probs[pair] = float(prob)
    return pair_probs


def score_ppi_chunked_pairs(
    unique_pairs: list[Pair],
    embedding_dict: dict,
    model,
    device: str,
    batch_size: int,
    n_out: int = 1,
) -> dict[Pair, float]:
    """Score unique chunked PPI pairs in GPU batches."""
    pair_probs: dict[Pair, float] = {}
    with torch.inference_mode():
        for start in range(0, len(unique_pairs), batch_size):
            pairs = unique_pairs[start : start + batch_size]
            left = torch.stack([embedding_dict[left].float() for left, _ in pairs])
            right = torch.stack([embedding_dict[right].float() for _, right in pairs])
            batch = torch.cat([left, right], dim=1).to(device)
            masks = torch.cat([chunk_mask(left), chunk_mask(right)], dim=1).to(device)
            probs = _as_prob_list(model(batch, masks), n_out=n_out)
            for pair, prob in zip(pairs, probs):
                pair_probs[pair] = float(prob)
    return pair_probs


def score_chunked_pairs(
    unique_pairs: list[Pair],
    left_dict: dict,
    right_dict: dict,
    model,
    device: str,
    batch_size: int,
) -> dict[Pair, float]:
    """Score unique two-sided chunked pairs in GPU batches."""
    pair_probs: dict[Pair, float] = {}
    with torch.inference_mode():
        for start in range(0, len(unique_pairs), batch_size):
            pairs = unique_pairs[start : start + batch_size]
            left = torch.stack([left_dict[left].float() for left, _ in pairs])
            right = torch.stack([right_dict[right].float() for _, right in pairs])
            left_mask = chunk_mask(left).to(device)
            right_mask = chunk_mask(right).to(device)
            logits = model(left.to(device), right.to(device), left_mask, right_mask)
            probs = _as_prob_list(logits)
            for pair, prob in zip(pairs, probs):
                pair_probs[pair] = float(prob)
    return pair_probs


def apply_pair_probabilities(
    results: list[dict],
    pairs_by_row: list[Pair],
    pair_probs: dict[Pair, float],
) -> None:
    """Mutate result rows with rounded probability and binary prediction."""
    for idx, pair in enumerate(pairs_by_row):
        prob = pair_probs.get(pair)
        if prob is None:
            continue
        rounded = round(prob, 4)
        results[idx]["probability"] = rounded
        results[idx]["prediction"] = 1 if prob >= 0.5 else 0
