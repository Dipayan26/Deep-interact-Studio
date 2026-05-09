from __future__ import annotations

import random

import pandas as pd


def _train_validation_indices(labels: list[int], test_size: float = 0.2, random_state: int = 42) -> tuple[list[int], list[int]]:
    idx = list(range(len(labels)))
    if len(idx) < 2:
        return idx, []

    rng = random.Random(random_state)
    by_label: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        by_label.setdefault(label, []).append(i)

    if len(by_label) > 1 and all(len(values) >= 2 for values in by_label.values()):
        train_idx: list[int] = []
        val_idx: list[int] = []
        for values in by_label.values():
            shuffled = values.copy()
            rng.shuffle(shuffled)
            n_val = min(len(shuffled) - 1, max(1, round(len(shuffled) * test_size)))
            val_idx.extend(shuffled[:n_val])
            train_idx.extend(shuffled[n_val:])
        return train_idx, val_idx

    shuffled = idx.copy()
    rng.shuffle(shuffled)
    n_val = min(len(shuffled) - 1, max(1, round(len(shuffled) * test_size)))
    return shuffled[n_val:], shuffled[:n_val]


def leakage_warnings(task_type: str, df: pd.DataFrame) -> list[str]:
    """
    Lightweight leakage-risk checks on mapped training data.
    Returns human-readable warnings; never raises for bad input.
    """
    try:
        if df.empty:
            return []

        warnings: list[str] = []
        if "label" in df.columns:
            labels = df["label"].astype(float).astype(int).tolist()
        else:
            labels = [0] * len(df)

        dup_rows = int(df.duplicated().sum())
        if dup_rows > 0:
            warnings.append(
                f"Detected {dup_rows} exact duplicate row(s). Duplicates can leak signals into validation."
            )

        tr_idx, va_idx = _train_validation_indices(labels, test_size=0.2, random_state=42)

        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[va_idx].reset_index(drop=True)

        def _ratio(a: set, b: set) -> float:
            return len(a & b) / max(1, len(a | b))

        if task_type == "ppi" and {"proteinA", "proteinB"}.issubset(df.columns):
            def _norm_seq(s):
                return str(s).strip().upper()

            p_a = df["proteinA"].map(_norm_seq)
            p_b = df["proteinB"].map(_norm_seq)

            pair_key_all = p_a.combine(p_b, lambda a, b: "|".join(sorted((a, b))))
            rev_dups = int(pair_key_all.duplicated().sum())
            if rev_dups > 0:
                warnings.append(
                    f"Detected {rev_dups} duplicate/reverse PPI pair(s) (A,B)/(B,A). "
                    "These can cause train/validation leakage."
                )

            tr_keys = set(
                tr_df["proteinA"].map(_norm_seq).combine(
                    tr_df["proteinB"].map(_norm_seq), lambda a, b: "|".join(sorted((a, b)))
                )
            )
            va_keys = set(
                va_df["proteinA"].map(_norm_seq).combine(
                    va_df["proteinB"].map(_norm_seq), lambda a, b: "|".join(sorted((a, b)))
                )
            )
            pair_overlap = len(tr_keys & va_keys)
            if pair_overlap > 0:
                warnings.append(
                    f"{pair_overlap} pair key(s) appear in both train and validation under random split. "
                    "Deduplicate pairs before training."
                )

            tr_entities = set(tr_df["proteinA"].map(_norm_seq)) | set(tr_df["proteinB"].map(_norm_seq))
            va_entities = set(va_df["proteinA"].map(_norm_seq)) | set(va_df["proteinB"].map(_norm_seq))
            ent_overlap = _ratio(tr_entities, va_entities)
            if ent_overlap >= 0.30:
                warnings.append(
                    f"High protein overlap between train/validation (~{ent_overlap*100:.1f}% Jaccard). "
                    "For stricter generalization, consider protein-disjoint splitting."
                )

        elif task_type == "dtpi" and {"smiles", "sequence"}.issubset(df.columns):
            def _norm_smiles(s):
                return str(s).strip()

            def _norm_seq(s):
                return str(s).strip().upper()

            pair_key_all = df["smiles"].map(_norm_smiles) + "|" + df["sequence"].map(_norm_seq)
            pair_dups = int(pair_key_all.duplicated().sum())
            if pair_dups > 0:
                warnings.append(
                    f"Detected {pair_dups} duplicate DTPI pair row(s). Duplicates can leak into validation."
                )

            tr_smiles = set(tr_df["smiles"].map(_norm_smiles))
            va_smiles = set(va_df["smiles"].map(_norm_smiles))
            tr_prots = set(tr_df["sequence"].map(_norm_seq))
            va_prots = set(va_df["sequence"].map(_norm_seq))
            sm_overlap = _ratio(tr_smiles, va_smiles)
            pr_overlap = _ratio(tr_prots, va_prots)
            if sm_overlap >= 0.30 or pr_overlap >= 0.30:
                warnings.append(
                    f"High entity overlap (SMILES ~{sm_overlap*100:.1f}%, proteins ~{pr_overlap*100:.1f}% Jaccard). "
                    "Consider scaffold/protein-disjoint split for stricter evaluation."
                )

        elif task_type == "rpi" and {"rna_sequence", "protein_sequence"}.issubset(df.columns):
            def _norm_rna(s):
                return str(s).strip().upper().replace("T", "U")

            def _norm_seq(s):
                return str(s).strip().upper()

            pair_key_all = df["rna_sequence"].map(_norm_rna) + "|" + df["protein_sequence"].map(_norm_seq)
            pair_dups = int(pair_key_all.duplicated().sum())
            if pair_dups > 0:
                warnings.append(
                    f"Detected {pair_dups} duplicate RPI pair row(s). Duplicates can leak into validation."
                )

            tr_rna = set(tr_df["rna_sequence"].map(_norm_rna))
            va_rna = set(va_df["rna_sequence"].map(_norm_rna))
            tr_prot = set(tr_df["protein_sequence"].map(_norm_seq))
            va_prot = set(va_df["protein_sequence"].map(_norm_seq))
            rna_overlap = _ratio(tr_rna, va_rna)
            prot_overlap = _ratio(tr_prot, va_prot)
            if rna_overlap >= 0.30 or prot_overlap >= 0.30:
                warnings.append(
                    f"High entity overlap (RNA ~{rna_overlap*100:.1f}%, proteins ~{prot_overlap*100:.1f}% Jaccard). "
                    "Consider RNA/protein-disjoint split for stricter evaluation."
                )

        elif task_type == "pdi" and {"dna_sequence", "protein_sequence"}.issubset(df.columns):
            def _norm_dna(s):
                return str(s).strip().upper()

            def _norm_seq(s):
                return str(s).strip().upper()

            pair_key_all = df["dna_sequence"].map(_norm_dna) + "|" + df["protein_sequence"].map(_norm_seq)
            pair_dups = int(pair_key_all.duplicated().sum())
            if pair_dups > 0:
                warnings.append(
                    f"Detected {pair_dups} duplicate PDI pair row(s). Duplicates can leak into validation."
                )

            tr_dna = set(tr_df["dna_sequence"].map(_norm_dna))
            va_dna = set(va_df["dna_sequence"].map(_norm_dna))
            tr_prot = set(tr_df["protein_sequence"].map(_norm_seq))
            va_prot = set(va_df["protein_sequence"].map(_norm_seq))
            dna_overlap = _ratio(tr_dna, va_dna)
            prot_overlap = _ratio(tr_prot, va_prot)
            if dna_overlap >= 0.30 or prot_overlap >= 0.30:
                warnings.append(
                    f"High entity overlap (DNA ~{dna_overlap*100:.1f}%, proteins ~{prot_overlap*100:.1f}% Jaccard). "
                    "Consider DNA/protein-disjoint split for stricter evaluation."
                )

        return warnings
    except Exception as exc:
        return [f"Leakage check could not be completed: {exc}"]


def mapped_training_frame(df: pd.DataFrame, columns: list[str], canonical_columns: list[str]) -> pd.DataFrame:
    mapped = df[columns].copy()
    mapped.columns = canonical_columns
    if "label" in mapped.columns:
        mapped["label"] = mapped["label"].astype(float).astype(int)
    return mapped
