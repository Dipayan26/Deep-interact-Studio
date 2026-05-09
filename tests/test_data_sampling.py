import pandas as pd

from frontend.data_sampling import (
    balanced_sample_by_label,
    compute_balanced_sample_counts,
    compute_label_sample_counts,
    sample_by_label_counts,
)


def test_balanced_sample_keeps_both_classes_from_sorted_input():
    df = pd.DataFrame({
        "pair_id": range(10000),
        "label": [1] * 5000 + [0] * 5000,
    })

    sampled, counts = balanced_sample_by_label(df, "label", 3000, 50, random_state=42)

    assert counts.selected_total == 3000
    assert counts.selected_pos == 1500
    assert counts.selected_neg == 1500
    assert sampled["label"].value_counts().to_dict() == {1: 1500, 0: 1500}


def test_balanced_sample_honors_requested_class_percent():
    counts = compute_balanced_sample_counts(
        n_pairs=1000,
        n_pos_available=1000,
        n_neg_available=1000,
        pos_percent=75,
    )

    assert counts.requested_pos == 750
    assert counts.requested_neg == 250
    assert counts.selected_pos == 750
    assert counts.selected_neg == 250


def test_balanced_sample_clamps_short_class_and_reduces_total():
    df = pd.DataFrame({
        "pair_id": range(5100),
        "label": [1] * 100 + [0] * 5000,
    })

    sampled, counts = balanced_sample_by_label(df, "label", 3000, 50, random_state=42)

    assert counts.selected_total == 1600
    assert counts.selected_pos == 100
    assert counts.selected_neg == 1500
    assert sampled["label"].value_counts().to_dict() == {0: 1500, 1: 100}


def test_count_sample_honors_requested_class_counts():
    counts = compute_label_sample_counts(
        n_pos_requested=40,
        n_neg_requested=60,
        n_pos_available=50,
        n_neg_available=80,
    )

    assert counts.requested_total == 100
    assert counts.requested_pos == 40
    assert counts.requested_neg == 60
    assert counts.selected_pos == 40
    assert counts.selected_neg == 60


def test_count_sample_selects_exact_positive_and_negative_counts():
    df = pd.DataFrame({
        "pair_id": range(100),
        "label": [1] * 50 + [0] * 50,
    })

    sampled, counts = sample_by_label_counts(df, "label", 25, 35, random_state=42)

    assert counts.selected_total == 60
    assert counts.selected_pos == 25
    assert counts.selected_neg == 35
    assert sampled["label"].value_counts().to_dict() == {0: 35, 1: 25}
