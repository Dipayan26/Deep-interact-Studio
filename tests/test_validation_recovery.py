import re

import pandas as pd

from frontend.validation_recovery import (
    invalid_embedding_row_mask,
    long_sequence_row_mask,
    trim_sequence_columns,
)


VALID_AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBJOUXZ*\-]+$")
VALID_RNA = re.compile(r"^[AUGCNaugcn]+$")
VALID_DNA = re.compile(r"^[ATGCNatgcn]+$")


def test_long_sequence_row_mask_flags_rows_with_either_long_protein():
    df = pd.DataFrame({
        "proteinA": ["A" * 10, "C" * 513, "D" * 20],
        "proteinB": ["E" * 512, "F" * 15, "G" * 600],
        "label": [1, 0, 1],
    })

    mask = long_sequence_row_mask(df, ["proteinA", "proteinB"], 512)

    assert mask.tolist() == [False, True, True]


def test_trim_sequence_columns_trims_and_normalizes_sequences_only():
    df = pd.DataFrame({
        "proteinA": [" acdef ", "c" * 515],
        "proteinB": ["m" * 513, "QRST"],
        "label": [1, 0],
        "note": ["keep", "also keep"],
    })

    trimmed = trim_sequence_columns(df, ["proteinA", "proteinB"], 512)

    assert trimmed.loc[0, "proteinA"] == "ACDEF"
    assert trimmed.loc[0, "proteinB"] == "M" * 512
    assert trimmed.loc[1, "proteinA"] == "C" * 512
    assert trimmed.loc[1, "proteinB"] == "QRST"
    assert trimmed["label"].tolist() == [1, 0]
    assert trimmed["note"].tolist() == ["keep", "also keep"]


def test_invalid_embedding_row_mask_checks_rows_beyond_preview_sample():
    df = pd.DataFrame({
        "proteinA": ["ACDE"] * 201,
        "proteinB": ["MNPQ"] * 201,
        "label": [1] * 201,
    })
    df.loc[200, "proteinB"] = "MNPQ!"

    mask = invalid_embedding_row_mask(
        df,
        {
            "proteinA": lambda value: bool(VALID_AA.match(value.upper())),
            "proteinB": lambda value: bool(VALID_AA.match(value.upper())),
        },
    )

    assert int(mask.sum()) == 1
    assert bool(mask.iloc[200])


def test_invalid_embedding_row_mask_flags_internal_whitespace():
    df = pd.DataFrame({
        "proteinA": ["AC DE", "ACDE"],
        "proteinB": ["MNPQ", "MN\tPQ"],
        "label": [1, 0],
    })

    mask = invalid_embedding_row_mask(
        df,
        {
            "proteinA": lambda value: bool(VALID_AA.match(value.upper())),
            "proteinB": lambda value: bool(VALID_AA.match(value.upper())),
        },
    )

    assert mask.tolist() == [True, True]


def test_invalid_embedding_row_mask_accepts_rna_t_as_u_and_n():
    df = pd.DataFrame({
        "rna": ["ACGTN", "AUGCN"],
        "protein": ["ACDE", "MNPQ"],
    })

    mask = invalid_embedding_row_mask(
        df,
        {
            "rna": lambda value: bool(VALID_RNA.match(value.upper().replace("T", "U"))),
            "protein": lambda value: bool(VALID_AA.match(value.upper())),
        },
    )

    assert mask.tolist() == [False, False]


def test_invalid_embedding_row_mask_accepts_dna_n():
    df = pd.DataFrame({
        "dna": ["ACGTN", "ACGTU"],
        "protein": ["ACDE", "MNPQ"],
    })

    mask = invalid_embedding_row_mask(
        df,
        {
            "dna": lambda value: bool(VALID_DNA.match(value.upper())),
            "protein": lambda value: bool(VALID_AA.match(value.upper())),
        },
    )

    assert mask.tolist() == [False, True]
