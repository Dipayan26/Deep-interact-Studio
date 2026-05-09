from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BalancedSampleCounts:
    requested_total: int
    requested_pos: int
    requested_neg: int
    selected_pos: int
    selected_neg: int

    @property
    def selected_total(self) -> int:
        return self.selected_pos + self.selected_neg


def compute_balanced_sample_counts(
    n_pairs: int,
    n_pos_available: int,
    n_neg_available: int,
    pos_percent: int,
) -> BalancedSampleCounts:
    requested_total = max(0, int(n_pairs))
    pos_percent = max(0, min(100, int(pos_percent)))

    requested_pos = int(round(requested_total * pos_percent / 100))
    requested_neg = requested_total - requested_pos

    selected_pos = min(requested_pos, max(0, int(n_pos_available)))
    selected_neg = min(requested_neg, max(0, int(n_neg_available)))

    return BalancedSampleCounts(
        requested_total=requested_total,
        requested_pos=requested_pos,
        requested_neg=requested_neg,
        selected_pos=selected_pos,
        selected_neg=selected_neg,
    )


def compute_label_sample_counts(
    n_pos_requested: int,
    n_neg_requested: int,
    n_pos_available: int,
    n_neg_available: int,
) -> BalancedSampleCounts:
    requested_pos = max(0, int(n_pos_requested))
    requested_neg = max(0, int(n_neg_requested))

    selected_pos = min(requested_pos, max(0, int(n_pos_available)))
    selected_neg = min(requested_neg, max(0, int(n_neg_available)))

    return BalancedSampleCounts(
        requested_total=requested_pos + requested_neg,
        requested_pos=requested_pos,
        requested_neg=requested_neg,
        selected_pos=selected_pos,
        selected_neg=selected_neg,
    )


def balanced_sample_by_label(
    df: pd.DataFrame,
    label_col: str,
    n_pairs: int,
    pos_percent: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, BalancedSampleCounts]:
    labels = df[label_col].astype(float).astype(int)
    pos_df = df[labels == 1]
    neg_df = df[labels == 0]

    counts = compute_balanced_sample_counts(
        n_pairs=n_pairs,
        n_pos_available=len(pos_df),
        n_neg_available=len(neg_df),
        pos_percent=pos_percent,
    )

    sampled_parts = []
    if counts.selected_pos:
        sampled_parts.append(
            pos_df.sample(n=counts.selected_pos, random_state=random_state)
            if counts.selected_pos < len(pos_df)
            else pos_df.copy()
        )
    if counts.selected_neg:
        sampled_parts.append(
            neg_df.sample(n=counts.selected_neg, random_state=random_state + 1)
            if counts.selected_neg < len(neg_df)
            else neg_df.copy()
        )

    if not sampled_parts:
        return df.iloc[0:0].copy(), counts

    sampled = pd.concat(sampled_parts, axis=0)
    sampled = sampled.sample(frac=1, random_state=random_state + 2).reset_index(drop=True)
    return sampled, counts


def sample_by_label_counts(
    df: pd.DataFrame,
    label_col: str,
    n_pos: int,
    n_neg: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, BalancedSampleCounts]:
    labels = df[label_col].astype(float).astype(int)
    pos_df = df[labels == 1]
    neg_df = df[labels == 0]

    counts = compute_label_sample_counts(
        n_pos_requested=n_pos,
        n_neg_requested=n_neg,
        n_pos_available=len(pos_df),
        n_neg_available=len(neg_df),
    )

    sampled_parts = []
    if counts.selected_pos:
        sampled_parts.append(
            pos_df.sample(n=counts.selected_pos, random_state=random_state)
            if counts.selected_pos < len(pos_df)
            else pos_df.copy()
        )
    if counts.selected_neg:
        sampled_parts.append(
            neg_df.sample(n=counts.selected_neg, random_state=random_state + 1)
            if counts.selected_neg < len(neg_df)
            else neg_df.copy()
        )

    if not sampled_parts:
        return df.iloc[0:0].copy(), counts

    sampled = pd.concat(sampled_parts, axis=0)
    sampled = sampled.sample(frac=1, random_state=random_state + 2).reset_index(drop=True)
    return sampled, counts
