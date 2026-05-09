import pandas as pd

from frontend.leakage_checks import leakage_warnings, mapped_training_frame


def test_mapped_training_frame_renames_columns_and_normalizes_label():
    raw = pd.DataFrame({
        "left": ["AAA"],
        "right": ["CCC"],
        "class": ["1.0"],
    })

    mapped = mapped_training_frame(raw, ["left", "right", "class"], ["proteinA", "proteinB", "label"])

    assert mapped.columns.tolist() == ["proteinA", "proteinB", "label"]
    assert mapped["label"].tolist() == [1]


def test_ppi_leakage_warnings_detect_reverse_pair_duplicate():
    df = pd.DataFrame({
        "proteinA": ["AAAA", "CCCC", "GGGG", "TTTT"],
        "proteinB": ["CCCC", "AAAA", "TTTT", "GGGG"],
        "label": [1, 1, 0, 0],
    })

    warnings = leakage_warnings("ppi", df)

    assert any("duplicate/reverse PPI pair" in warning for warning in warnings)


def test_rpi_leakage_warnings_treat_t_and_u_as_same_rna():
    df = pd.DataFrame({
        "rna_sequence": ["ACGT", "ACGU", "UUUU", "CCCC"],
        "protein_sequence": ["MKK", "MKK", "AAA", "GGG"],
        "label": [1, 1, 0, 0],
    })

    warnings = leakage_warnings("rpi", df)

    assert any("duplicate RPI pair" in warning for warning in warnings)
