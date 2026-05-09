import random
from typing import Callable

import pandas as pd
import streamlit as st


VALID_LABELS = {"0", "1", "0.0", "1.0"}


def build_recoverable_row_mask(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    col_label: str,
    is_valid_a: Callable[[str], bool],
    is_valid_b: Callable[[str], bool],
) -> pd.Series:
    mask = pd.Series(False, index=df.index)

    for col in [col_a, col_b, col_label]:
        mask |= df[col].isnull()

    for col in [col_a, col_b]:
        mask |= df[col].astype(str).str.strip().eq("")

    for col, validator in [(col_a, is_valid_a), (col_b, is_valid_b)]:
        values = df[col].astype(str).str.strip()
        mask |= values.apply(lambda value: not validator(value))

    raw_labels = df[col_label].astype(str).str.strip()
    mask |= ~raw_labels.isin(VALID_LABELS)
    return mask


def make_cleaned_df(df: pd.DataFrame, affected_mask: pd.Series) -> pd.DataFrame:
    return df.loc[~affected_mask].copy().reset_index(drop=True)


def long_sequence_row_mask(
    df: pd.DataFrame,
    sequence_cols: list[str],
    max_len: int,
) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in sequence_cols:
        lengths = df[col].astype(str).str.strip().str.len()
        mask |= lengths.gt(max_len)
    return mask


def trim_sequence_columns(
    df: pd.DataFrame,
    sequence_cols: list[str],
    max_len: int,
    uppercase: bool = True,
) -> pd.DataFrame:
    trimmed = df.copy()
    for col in sequence_cols:
        values = trimmed[col].astype(str).str.strip()
        if uppercase:
            values = values.str.upper()
        trimmed[col] = values.str.slice(0, max_len)
    return trimmed.reset_index(drop=True)


def invalid_embedding_row_mask(
    df: pd.DataFrame,
    validators: dict[str, Callable[[str], bool]],
) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col, validator in validators.items():
        values = df[col].astype(str).str.strip()
        mask |= values.apply(lambda value: not validator(value))
    return mask


def render_invalid_embedding_cleanup(
    df: pd.DataFrame,
    invalid_mask: pd.Series,
    preview_cols: list[str],
    edited_key: str,
    edited_flag_key: str,
    key_prefix: str,
    reason: str = "invalid characters or internal whitespace that can prevent reliable embedding",
) -> bool:
    invalid_count = int(invalid_mask.sum())
    if invalid_count == 0:
        return False

    st.warning(
        f"{invalid_count:,} row(s) contain {reason}. "
        "Remove these rows before configuring data sampling."
    )

    preview = df.loc[invalid_mask, preview_cols].head(5).copy()
    preview.insert(0, "source row", preview.index.to_series().astype(int) + 2)
    for col in preview_cols:
        preview[col] = preview[col].astype(str).str[:50] + "..."
    st.dataframe(preview, use_container_width=True, hide_index=True)

    if st.button(
        f"Remove {invalid_count:,} invalid embedding row(s)",
        key=f"{key_prefix}_remove_invalid_embedding_rows",
        use_container_width=True,
    ):
        cleaned = df.loc[~invalid_mask].copy().reset_index(drop=True)
        if cleaned.empty:
            st.error("Removing invalid rows would remove all rows. Fix the source CSV and upload it again.")
        else:
            apply_edited_df(cleaned, edited_key, edited_flag_key)

    return True


def can_generate_negatives(df: pd.DataFrame, col_label: str) -> bool:
    raw = df[col_label].astype(str).str.strip()
    if (~raw.isin(VALID_LABELS)).any():
        return False
    labels = raw.astype(float).astype(int)
    return bool((labels == 1).any() and not (labels == 0).any())


def generate_negative_samples(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    col_label: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, int, int]:
    positives = df.copy().reset_index(drop=True)
    raw = positives[col_label].astype(str).str.strip()
    positives = positives.loc[raw.astype(float).astype(int) == 1].copy().reset_index(drop=True)
    if positives.empty:
        return df.copy(), 0, 0

    target_count = len(positives)
    values_a = positives[col_a].astype(str).tolist()
    values_b = positives[col_b].astype(str).tolist()
    original_pairs = set(zip(values_a, values_b))
    generated_pairs: set[tuple[str, str]] = set()
    generated_rows = []
    rng = random.Random(random_state)

    max_attempts = max(1000, target_count * 50)
    for _ in range(max_attempts):
        if len(generated_rows) >= target_count:
            break

        value_a = rng.choice(values_a)
        value_b = rng.choice(values_b)
        pair = (value_a, value_b)
        if pair in original_pairs or pair in generated_pairs:
            continue

        row = positives.iloc[rng.randrange(len(positives))].copy()
        row[col_a] = value_a
        row[col_b] = value_b
        row[col_label] = 0
        generated_rows.append(row)
        generated_pairs.add(pair)

    if not generated_rows:
        return df.copy(), 0, target_count

    negatives = pd.DataFrame(generated_rows, columns=df.columns)
    combined = pd.concat([positives, negatives], ignore_index=True)
    combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return combined, len(negatives), target_count


def apply_edited_df(df: pd.DataFrame, edited_key: str, edited_flag_key: str) -> None:
    st.session_state[edited_key] = df.copy().reset_index(drop=True)
    st.session_state[edited_flag_key] = True
    st.rerun()


def clear_edited_df(edited_key: str, edited_flag_key: str) -> None:
    st.session_state.pop(edited_key, None)
    st.session_state[edited_flag_key] = False


def render_edited_download(
    df: pd.DataFrame,
    edited_flag_key: str,
    file_name: str,
) -> None:
    if not st.session_state.get(edited_flag_key, False):
        return

    st.download_button(
        "Download edited CSV",
        data=df.to_csv(index=False),
        file_name=file_name,
        mime="text/csv",
        help="Download the cleaned or augmented training data currently used by this page.",
        use_container_width=True,
    )


def render_recovery_controls(
    df: pd.DataFrame,
    affected_mask: pd.Series,
    col_a: str,
    col_b: str,
    col_label: str,
    edited_key: str,
    edited_flag_key: str,
    key_prefix: str,
) -> None:
    affected_count = int(affected_mask.sum())

    if affected_count:
        if st.button(
            f"Remove {affected_count:,} affected row(s)",
            key=f"{key_prefix}_remove_affected_rows",
            use_container_width=True,
        ):
            cleaned = make_cleaned_df(df, affected_mask)
            if cleaned.empty:
                st.error("Cleaning would remove all rows. Fix the source CSV and upload it again.")
            else:
                apply_edited_df(cleaned, edited_key, edited_flag_key)

    if affected_count == 0 and can_generate_negatives(df, col_label):
        if st.button(
            "Generate negative samples",
            key=f"{key_prefix}_generate_negatives",
            use_container_width=True,
        ):
            augmented, generated_count, target_count = generate_negative_samples(
                df, col_a, col_b, col_label
            )
            if generated_count == 0:
                st.error("Could not generate any unique negative pairs from this input.")
            else:
                if generated_count < target_count:
                    st.warning(
                        f"Generated {generated_count:,} of {target_count:,} requested negative pairs "
                        "because there were not enough unique random pairings."
                    )
                apply_edited_df(augmented, edited_key, edited_flag_key)
