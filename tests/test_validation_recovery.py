import pandas as pd

from frontend.validation_recovery import long_sequence_row_mask, trim_sequence_columns


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
