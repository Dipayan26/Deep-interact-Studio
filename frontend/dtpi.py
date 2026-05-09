import io
import json
import math
import os
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from architecture_graph import render_architecture_graph
from data_sampling import compute_label_sample_counts, sample_by_label_counts
from leakage_checks import leakage_warnings, mapped_training_frame
from model_builder_defaults import default_layers, reset_model_builder_state
from validation_recovery import (
    apply_edited_df,
    build_recoverable_row_mask,
    clear_edited_df,
    invalid_embedding_row_mask,
    long_sequence_row_mask,
    render_edited_download,
    render_invalid_embedding_cleanup,
    render_recovery_controls,
    trim_sequence_columns,
)

BACKEND = os.getenv("BACKEND_URL", "http://backend:8005")
MAX_MODEL_PARAMS = 5_000_000
MAX_DTPI_RESIDUES = 512

_WORKFLOW_STEPS = ["Data", "Architecture", "Training"]
_DTPI_STEP_THEMES = {
    "Data": {"color": "#16705D", "bg": "#BCE7DB", "soft": "#CFF3E8"},
    "Architecture": {"color": "#16705D", "bg": "#C5E6DD", "soft": "#CFF3E8"},
    "Training": {"color": "#16705D", "bg": "#BBE3D8", "soft": "#CFF3E8"},
}


_STEP_KEY = "dtpi_builder_step"
_STEP_SELECTOR_KEY = "dtpi_builder_step_selector"
_DATA_CONTEXT_KEY = "dtpi_data_context"


def _set_builder_step(step: str) -> None:
    st.session_state[_STEP_KEY] = step
    st.rerun()


def _render_step_nav(active_step: str, next_disabled: bool = False) -> None:
    current_idx = _WORKFLOW_STEPS.index(active_step)
    nav_cols = st.columns([1, 1, 4])
    with nav_cols[0]:
        if st.button(
            "Previous",
            disabled=current_idx == 0,
            use_container_width=True,
            key=f"dtpi_prev_{active_step.lower()}",
        ):
            _set_builder_step(_WORKFLOW_STEPS[current_idx - 1])
    with nav_cols[1]:
        if st.button(
            "Next",
            disabled=current_idx == len(_WORKFLOW_STEPS) - 1 or next_disabled,
            use_container_width=True,
            key=f"dtpi_next_{active_step.lower()}",
        ):
            _set_builder_step(_WORKFLOW_STEPS[current_idx + 1])


def _render_workflow_band(step: str) -> None:
    theme = _DTPI_STEP_THEMES[step]
    step_number = _WORKFLOW_STEPS.index(step) + 1
    st.markdown(
        f"""
        <style>
        :root {{
            --workflow-step-color: {theme["color"]};
            --workflow-step-bg: {theme["bg"]};
            --workflow-step-soft: {theme["soft"]};
        }}
        </style>
        <div class="workflow-step-band">
            <div class="workflow-step-kicker">Step {step_number} of {len(_WORKFLOW_STEPS)}</div>
            <div class="workflow-step-title">{step}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

_VALID_AA    = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBJOUXZ*\-]+$")
_VALID_SMILES = re.compile(r"^[A-Za-z0-9@+\-\[\]()\=\#\%\\/\.\*~:\s]+$")

# ── Embedding model options ───────────────────────────────────────────────────
CHEMBERTA_OPTIONS = {
    "ChemBERTa-zinc-base (768-dim, default)": ("seyonec/ChemBERTa-zinc-base-v1", 768),
}

ESM2_OPTIONS = {
    "ESM2 8M  (320-dim, fastest)":   ("esm2_t6_8M_UR50D",    320),
    "ESM2 35M (480-dim, default)":   ("esm2_t12_35M_UR50D",  480),
    "ESM2 150M (640-dim, accurate)": ("esm2_t30_150M_UR50D", 640),
}

# ── Demo data generator ───────────────────────────────────────────────────────

def _generate_demo_csv() -> str:
    """Generate synthetic DTPI demo data (50 pairs) with a fixed random seed."""
    rng = random.Random(7)

    AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
    AA_WEIGHTS  = [
        0.0669, 0.0166, 0.0545, 0.0672, 0.0386,
        0.0709, 0.0228, 0.0580, 0.0589, 0.0966,
        0.0242, 0.0406, 0.0472, 0.0393, 0.0553,
        0.0664, 0.0534, 0.0686, 0.0109, 0.0292,
    ]

    # Synthetic SMILES fragments to combine
    _SMILES_FRAGMENTS = [
        "CC(=O)O", "c1ccccc1", "CC(N)=O", "CC(=O)Nc1ccc(O)cc1",
        "C1CCCCC1", "c1ccncc1", "CC(C)CC", "OCC(O)CO",
        "Cc1ccc(O)cc1", "CC(=O)c1ccc(N)cc1",
    ]

    def _rand_smiles() -> str:
        parts = rng.sample(_SMILES_FRAGMENTS, k=rng.randint(1, 3))
        return ".".join(parts)

    def _rand_seq(length: int) -> str:
        return "".join(rng.choices(AA_ALPHABET, weights=AA_WEIGHTS, k=length))

    rows = []
    used = set()
    for label in ([1] * 25 + [0] * 25):
        for _ in range(300):
            smi = _rand_smiles()
            seq = _rand_seq(rng.randint(60, 150))
            key = (smi, seq[:20])
            if key not in used:
                used.add(key)
                rows.append({"smiles": smi, "sequence": seq, "label": label})
                break

    rng.shuffle(rows)
    lines = ["smiles,sequence,label"]
    for r in rows:
        lines.append(f"{r['smiles']},{r['sequence']},{r['label']}")
    return "\n".join(lines) + "\n"


_DEMO_CSV = _generate_demo_csv()


# ── Architecture dimension calculator ─────────────────────────────────────────

def _compute_out_dim(layer_type: str, in_dim: int, cfg: dict) -> int:
    lt = layer_type.lower()
    if lt == "linear":
        return int(cfg.get("hidden_dim", 256))
    if lt == "cnn1d":
        return int(cfg.get("out_channels", 64))
    if lt == "bilstm":
        return 2 * int(cfg.get("hidden_size", 128))
    if lt == "gru":
        hidden = int(cfg.get("hidden_size", 128))
        bidir  = bool(cfg.get("bidirectional", True))
        return 2 * hidden if bidir else hidden
    if lt == "transformer":
        return int(cfg.get("d_model", 256))
    if lt == "residual":
        return in_dim
    return in_dim


def _total_param_count(input_dim: int, layer_configs: list, sequence_mode: bool = False, projection_dims: tuple[int, int] | None = None) -> int:
    total = 0
    cur   = input_dim
    if sequence_mode and projection_dims is not None:
        left_dim, right_dim = projection_dims
        total += left_dim * input_dim + input_dim
        total += right_dim * input_dim + input_dim
        total += 2 * input_dim
    for cfg in layer_configs:
        lt = cfg["type"].lower()
        if lt == "linear":
            h = int(cfg.get("hidden_dim", 256))
            total += cur * h + h
            if cfg.get("batchnorm"):
                total += 2 * h
            cur = h
        elif lt == "cnn1d":
            out_ch = int(cfg.get("out_channels", 64))
            k      = int(cfg.get("kernel_size", 3))
            in_ch = cur if sequence_mode else 1
            total += in_ch * out_ch * k + out_ch
            cur = out_ch
        elif lt == "bilstm":
            h    = int(cfg.get("hidden_size", 128))
            nl   = int(cfg.get("num_layers", 1))
            gate = 4
            dirs = 2
            total += dirs * gate * (cur * h + h * h + 2 * h)
            for _ in range(nl - 1):
                total += dirs * gate * (dirs * h * h + h * h + 2 * h)
            cur = dirs * h
        elif lt == "gru":
            h     = int(cfg.get("hidden_size", 128))
            nl    = int(cfg.get("num_layers", 1))
            bidir = bool(cfg.get("bidirectional", True))
            gate  = 3
            dirs  = 2 if bidir else 1
            total += dirs * gate * (cur * h + h * h + 2 * h)
            for _ in range(nl - 1):
                total += dirs * gate * (dirs * h * h + h * h + 2 * h)
            cur = dirs * h
        elif lt == "transformer":
            d   = int(cfg.get("d_model", 256))
            ff  = int(cfg.get("dim_feedforward", d * 2))
            nl  = int(cfg.get("num_layers", 2))
            total += cur * d + d
            total += nl * (4 * d * d + 4 * d + d * ff + ff + ff * d + d + 4 * d)
            cur = d
        elif lt == "residual":
            h = int(cfg.get("hidden_dim", 256))
            total += cur * h + h + h * cur + cur
            if cfg.get("batchnorm"):
                total += 2 * h
            total += 2 * cur
    head_in = 2 * cur if sequence_mode else cur
    total += head_in * 1 + 1
    return total


# ── Architecture figure ───────────────────────────────────────────────────────

def _arch_figure(layer_configs: list, chem_dim: int, prot_dim: int) -> plt.Figure:
    BG        = "#EEF5F4"  # slight teal tint vs PPI's warm beige
    input_dim = chem_dim + prot_dim

    entries = [{
        "name": "Input",
        "dim":  input_dim,
        "sub":  f"concat(chem,prot)\n{input_dim}-dim",
    }]
    cur_dim = input_dim
    for i, cfg in enumerate(layer_configs):
        lt      = cfg["type"]
        out_dim = _compute_out_dim(lt, cur_dim, cfg)
        entries.append({"name": f"Layer {i + 1}", "dim": out_dim, "sub": lt.upper()})
        cur_dim = out_dim
    entries.append({"name": "Output", "dim": 1, "sub": "sigmoid(logit)"})

    n    = len(entries)
    COLS = ["#2A7D6F"] + ["#1E6860"] * (n - 2) + ["#144D47"]

    fig_w   = max(4.2, n * 1.28)
    fig, ax = plt.subplots(figsize=(fig_w, 2.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    max_dim  = max(e["dim"] for e in entries)
    MAX_H, MIN_H = 1.85, 0.34
    xs    = [(i + 0.5) * (fig_w / n) for i in range(n)]
    box_w = (fig_w / n) * 0.52
    cy    = 1.45

    for i, (entry, x, col) in enumerate(zip(entries, xs, COLS)):
        h  = MIN_H + (MAX_H - MIN_H) * (
            math.log2(entry["dim"] + 1) / math.log2(max_dim + 1)
        )
        y0 = cy - h / 2

        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y0), box_w, h,
            boxstyle="round,pad=0.05",
            facecolor=col, edgecolor="white", linewidth=1.2,
        )
        ax.add_patch(rect)

        ax.text(x, cy, f"{entry['dim']:,}",
                ha="center", va="center",
                color="white", fontsize=8, fontweight="bold",
                fontfamily="monospace")

        ax.text(x, y0 + h + 0.1, entry["name"],
                ha="center", va="bottom",
                color="#1C1C1C", fontsize=7)

        ax.text(x, y0 - 0.1, entry["sub"],
                ha="center", va="top",
                color="#666666", fontsize=6, style="italic")

        if i < n - 1:
            x_next = xs[i + 1]
            ax.annotate(
                "", xy=(x_next - box_w / 2 - 0.06, cy),
                xytext=(x + box_w / 2 + 0.06, cy),
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.3),
            )

    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, 2.9)
    plt.tight_layout(pad=0.2)
    return fig


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(df, col_smiles, col_seq, col_label):
    errors, warnings = [], []

    for col in [col_smiles, col_seq, col_label]:
        n = df[col].isnull().sum()
        if n:
            errors.append(f"Column **{col}** has {n} missing value(s).")
    if errors:
        return errors, warnings, {}

    # SMILES: basic character check
    bad_smi = df[col_smiles].astype(str).str.strip().apply(
        lambda s: not bool(_VALID_SMILES.match(s)) or len(s) == 0
    )
    if bad_smi.sum():
        first = df[col_smiles][bad_smi].iloc[0][:40]
        errors.append(
            f"Column **{col_smiles}** contains {int(bad_smi.sum())} invalid SMILES string(s) "
            f"(first: `{first}...`)."
        )

    # Protein sequence: amino acid alphabet
    sample = df[col_seq].head(200)
    bad_seq = sample.astype(str).str.strip().str.upper().apply(
        lambda s: not bool(_VALID_AA.match(s)) or len(s) == 0
    )
    if bad_seq.sum():
        first = sample[bad_seq].iloc[0][:30]
        errors.append(
            f"Column **{col_seq}** contains {int(bad_seq.sum())} sequence(s) with invalid "
            f"characters (first: `{first}...`). Only amino acid letters accepted."
        )

    # Label check
    raw = df[col_label].astype(str).str.strip()
    if (~raw.isin({"0", "1", "0.0", "1.0"})).any():
        errors.append(
            f"Column **{col_label}** must contain only `0` and `1`. "
            f"Found: {raw[~raw.isin({'0','1','0.0','1.0'})].unique()[:5].tolist()}"
        )
    else:
        li   = raw.astype(float).astype(int)
        npos = int((li == 1).sum())
        nneg = int((li == 0).sum())
        if npos == 0:
            errors.append("No positive examples (label=1). Both classes are required.")
        if nneg == 0:
            errors.append("No negative examples (label=0). Both classes are required.")
        if npos > 0 and nneg > 0:
            r = npos / nneg
            if r > 9 or r < 1 / 9:
                warnings.append(
                    f"Class imbalance: {npos:,} positive vs {nneg:,} negative "
                    f"(ratio {r:.1f}:1). Weighted loss will compensate."
                )

    n_rows = len(df)
    if n_rows < 20:
        warnings.append(
            f"Only {n_rows} pairs. At least 100–200 pairs recommended for reliable training."
        )

    stats = {
        "rows":        n_rows,
        "unique_drugs": df[col_smiles].nunique(),
        "unique_seqs":  df[col_seq].nunique(),
        "n_pos": int((df[col_label].astype(float).astype(int) == 1).sum()) if not errors else 0,
        "n_neg": int((df[col_label].astype(float).astype(int) == 0).sum()) if not errors else 0,
    }
    return errors, warnings, stats


def _estimate_time(n_pairs: int, n_unique_seqs: int, mean_seq_len: float,
                   n_unique_drugs: int, epochs: int, batch_size: int) -> str:
    embed_s  = (n_unique_seqs / 8) * 0.25            # ESM2 embedding
    embed_s += n_unique_drugs * 0.05                  # ChemBERTa (faster)
    train_s  = epochs * math.ceil(n_pairs * 0.8 / batch_size) * 0.04
    total    = embed_s + train_s
    if total < 90:
        return f"~{int(total)} s"
    if total < 3600:
        return f"~{int(total / 60)} min"
    return f"~{total / 3600:.1f} h"


# =============================================================================
# Page
# =============================================================================

# ── DTPI teal theme ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Top accent bar */
[data-testid="stAppViewContainer"] > section:first-child::before {
    content: "";
    display: block;
    height: 4px;
    background: linear-gradient(90deg, #2A7D6F, #3AAFA9);
    margin-bottom: 0;
}
/* Subheader colour */
h2, h3 { color: #1a5c55 !important; }
/* Primary button */
div[data-testid="stButton"] button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background-color: #2A7D6F !important;
    border-color: #2A7D6F !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background-color: #1e5e52 !important;
    border-color: #1e5e52 !important;
}
/* Active tab / selected widget accent */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #2A7D6F !important;
}
div[role="radiogroup"] {
    gap: 0.5rem;
}
div[role="radiogroup"] label {
    border: 1px solid #D9E0E8;
    border-radius: 8px;
    padding: 0.42rem 0.75rem;
    background: #FFFFFF;
}
div[role="radiogroup"] label:has(input:checked) {
    border-color: var(--builder-step-color);
    background: var(--builder-step-bg);
    box-shadow: inset 0 0 0 1px var(--builder-step-color);
}
div[role="radiogroup"] label p {
    font-weight: 700;
    font-size: 1rem;
}
div[role="radiogroup"] label:has(input:checked) p {
    color: var(--builder-step-color);
}
.workflow-step-band {
    margin: 0.85rem 0 1.15rem;
    padding: 0.95rem 1.1rem;
    border: 1px solid color-mix(in srgb, var(--workflow-step-color) 34%, white);
    border-left: 8px solid var(--workflow-step-color);
    border-radius: 8px;
    background: linear-gradient(90deg, var(--workflow-step-bg), var(--workflow-step-soft));
    box-shadow: 0 6px 18px rgba(21, 37, 54, 0.08);
}
.workflow-step-kicker {
    color: var(--workflow-step-color);
    font-size: 0.82rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0;
}
.workflow-step-title {
    color: var(--workflow-step-color);
    font-size: 1.65rem;
    font-weight: 900;
    line-height: 1.15;
    margin-top: 0.18rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Drug-Target Protein Interaction")
st.caption("Train a classifier to predict binding between small molecules and target proteins.")
st.divider()

st.session_state.setdefault(_STEP_KEY, "Data")
if st.session_state[_STEP_KEY] not in _WORKFLOW_STEPS:
    st.session_state[_STEP_KEY] = "Data"
if st.session_state.get(_STEP_SELECTOR_KEY) != st.session_state[_STEP_KEY]:
    st.session_state[_STEP_SELECTOR_KEY] = st.session_state[_STEP_KEY]

active_step = st.radio(
    "Builder section",
    _WORKFLOW_STEPS,
    horizontal=True,
    key=_STEP_SELECTOR_KEY,
    label_visibility="collapsed",
)
st.session_state[_STEP_KEY] = active_step
builder_step_theme = _DTPI_STEP_THEMES[active_step]
st.markdown(
    f"""
    <style>
    :root {{
        --builder-step-color: {builder_step_theme["color"]};
        --builder-step-bg: {builder_step_theme["bg"]};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

data_context = st.session_state.get(_DATA_CONTEXT_KEY, {})
data_ready = bool(data_context.get("data_ready", False))
raw_df = data_context.get("raw_df")
col_smiles = data_context.get("col_smiles", "smiles")
col_seq = data_context.get("col_seq", "sequence")
col_label = data_context.get("col_label", "label")
mean_seq_len = float(data_context.get("mean_seq_len", 0.0))
stats = data_context.get("stats", {"rows": 0, "unique_drugs": 0, "unique_seqs": 0, "n_pos": 0, "n_neg": 0})
positive_pairs_use = int(data_context.get("positive_pairs_use", 0))
negative_pairs_use = int(data_context.get("negative_pairs_use", 0))
n_pairs_use = int(data_context.get("n_pairs_use", positive_pairs_use + negative_pairs_use))
train_split = int(data_context.get("train_split", 80))
sampling_over_cap = bool(data_context.get("sampling_over_cap", False))

default_chem_label = list(CHEMBERTA_OPTIONS.keys())[0]
chem_label = st.session_state.get("dtpi_chem_label", default_chem_label)
if chem_label not in CHEMBERTA_OPTIONS:
    chem_label = default_chem_label
chem_model_name, chem_dim = CHEMBERTA_OPTIONS[chem_label]
default_esm2_label = list(ESM2_OPTIONS.keys())[1]
esm2_label = st.session_state.get("dtpi_esm2_label", default_esm2_label)
if esm2_label not in ESM2_OPTIONS:
    esm2_label = default_esm2_label
esm_model_name, esm_dim = ESM2_OPTIONS[esm2_label]
embedding_representation = st.session_state.get("dtpi_embedding_representation", "pooled")
smiles_chunk_max_len = 512
protein_chunk_max_len = MAX_DTPI_RESIDUES
smiles_num_chunks = int(st.session_state.get("dtpi_smiles_chunks", 8))
protein_num_chunks = int(st.session_state.get("dtpi_protein_chunks", 8))
chunk_dtype = st.session_state.get("dtpi_chunk_dtype", "float16")
sequence_mode = embedding_representation == "chunked"
input_dim = max(chem_dim, esm_dim) if sequence_mode else chem_dim + esm_dim
st.session_state.setdefault("dtpi_layers", default_layers())
st.session_state.setdefault("_dtpi_lid", 2)
layers: list = st.session_state["dtpi_layers"]
layer_configs_display = [
    {k: v for k, v in lyr.items() if k != "id"}
    for lyr in layers
]
n_param = _total_param_count(
    input_dim,
    layer_configs_display,
    sequence_mode=sequence_mode,
    projection_dims=(chem_dim, esm_dim) if sequence_mode else None,
) if layers else 0
param_limit_exceeded = n_param > MAX_MODEL_PARAMS

if active_step == "Data":

        # ── 1. Training data ──────────────────────────────────────────────────────────
    _render_workflow_band("Data")
    st.subheader("Training Data")

    st.session_state.setdefault("dtpi_demo_loaded", False)
    st.session_state.setdefault("dtpi_uploader_key", 0)
    _EDITED_DF_KEY = "dtpi_edited_df"
    _EDITED_FLAG_KEY = "dtpi_data_edited"
    _SOURCE_KEY = "dtpi_data_source"
    _POS_COUNT_KEY = "dtpi_positive_count"
    _NEG_COUNT_KEY = "dtpi_negative_count"
    _SAMPLING_SIGNATURE_KEY = "dtpi_sampling_signature"
    st.session_state.setdefault(_EDITED_FLAG_KEY, False)

    col_dl, col_ex, _, col_reset = st.columns([2, 2, 2, 1])
    with col_dl:
        st.download_button(
            "Download demo CSV", data=_DEMO_CSV,
            file_name="demo_dtpi_train.csv", mime="text/csv",
            help="Download the synthetic demo CSV for offline inspection.",
            use_container_width=True,
        )
    with col_ex:
        if st.button("Load Example Data", help="Loads 50 synthetic DTPI pairs to test the workflow.",
                     use_container_width=True):
            clear_edited_df(_EDITED_DF_KEY, _EDITED_FLAG_KEY)
            st.session_state["dtpi_demo_loaded"] = True
            st.session_state["dtpi_uploader_key"] += 1
            st.rerun()
    with col_reset:
        if st.button("Reset", help="Clear loaded data and reset the model builder.", use_container_width=True):
            clear_edited_df(_EDITED_DF_KEY, _EDITED_FLAG_KEY)
            st.session_state["dtpi_demo_loaded"] = False
            st.session_state["dtpi_uploader_key"] += 1
            reset_model_builder_state(
                "dtpi_layers",
                "_dtpi_lid",
                widget_prefix="dtpi_",
                new_layer_type_key="dtpi_new_layer_type",
                model_defaults={
                    "dtpi_chem_label": list(CHEMBERTA_OPTIONS.keys())[0],
                    "dtpi_esm2_label": list(ESM2_OPTIONS.keys())[1],
                },
            )
            st.rerun()

    st.caption(
        "Upload a CSV with any column names. "
        "You will map them to **SMILES**, **Protein Sequence**, and **Label (0/1)** below."
    )

    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"], label_visibility="collapsed",
        key=f"dtpi_uploader_{st.session_state['dtpi_uploader_key']}",
    )

    demo_loaded = st.session_state.get("dtpi_demo_loaded", False)
    data_ready  = False
    raw_df      = None

    if uploaded is not None:
        try:
            source_id = (
                "upload",
                getattr(uploaded, "file_id", None),
                uploaded.name,
                getattr(uploaded, "size", None),
            )
            if st.session_state.get(_SOURCE_KEY) != source_id:
                clear_edited_df(_EDITED_DF_KEY, _EDITED_FLAG_KEY)
                st.session_state[_SOURCE_KEY] = source_id
            uploaded.seek(0)
            raw_df = pd.read_csv(uploaded)
            if _EDITED_DF_KEY in st.session_state:
                raw_df = st.session_state[_EDITED_DF_KEY].copy()
            data_ready = True
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()
    elif demo_loaded:
        source_id = ("demo",)
        if st.session_state.get(_SOURCE_KEY) != source_id:
            clear_edited_df(_EDITED_DF_KEY, _EDITED_FLAG_KEY)
            st.session_state[_SOURCE_KEY] = source_id
        st.info("Using example data (50 synthetic DTPI pairs). Upload your own CSV to override.")
        raw_df = pd.read_csv(io.StringIO(_DEMO_CSV))
        if _EDITED_DF_KEY in st.session_state:
            raw_df = st.session_state[_EDITED_DF_KEY].copy()
        data_ready = True

    if data_ready:
        if raw_df.empty:
            st.error("The file is empty.")
            st.stop()
        cols = raw_df.columns.tolist()
        st.markdown(f"**{len(raw_df):,} rows · {len(cols)} columns:** {', '.join(f'`{c}`' for c in cols)}")
        st.dataframe(
            raw_df.head(5).astype(str).apply(
                lambda s: s.str[:60] + "…" if s.str.len().max() > 60 else s
            ),
            use_container_width=True, hide_index=True,
        )
        render_edited_download(raw_df, _EDITED_FLAG_KEY, "edited_dtpi_train.csv")

    # ── Grey-out sections below when no data is loaded ───────────────────────────
    if not data_ready:
        st.markdown("""
        <style>
        div[data-testid="stVerticalBlock"] > div:has(.dtpi-grey-marker) ~ div {
            opacity: 0.4;
            pointer-events: none;
            user-select: none;
        }
        </style>
        <span class="dtpi-grey-marker"></span>
        """, unsafe_allow_html=True)

    st.divider()

    # ── 2. Column mapping ─────────────────────────────────────────────────────────
    st.subheader("Column Mapping")

    _placeholder_cols = ["smiles", "sequence", "label"]
    _sel_cols = raw_df.columns.tolist() if data_ready else _placeholder_cols

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        col_smiles = st.selectbox("SMILES",            _sel_cols, index=0)
    with mc2:
        col_seq    = st.selectbox("Protein Sequence",  _sel_cols, index=min(1, len(_sel_cols) - 1))
    with mc3:
        col_label  = st.selectbox("Label (0/1)",       _sel_cols, index=min(2, len(_sel_cols) - 1))

    mean_seq_len = 0.0
    stats        = {"rows": 0, "unique_drugs": 0, "unique_seqs": 0, "n_pos": 0, "n_neg": 0}
    n_pairs_use  = 0
    positive_pairs_use = 0
    negative_pairs_use = 0
    train_split  = 80
    sampling_over_cap = False

    if data_ready:
        if col_smiles == col_seq:
            st.error("SMILES and Protein Sequence cannot be the same column.")
            st.stop()

        errors, warnings, stats = _validate(raw_df, col_smiles, col_seq, col_label)
        affected_mask = build_recoverable_row_mask(
            raw_df,
            col_smiles,
            col_seq,
            col_label,
            lambda value: bool(_VALID_SMILES.match(value)) and len(value) > 0,
            lambda value: bool(_VALID_AA.match(value.upper())),
        )

        if errors:
            for e in errors:
                st.error(e)
            render_recovery_controls(
                raw_df,
                affected_mask,
                col_smiles,
                col_seq,
                col_label,
                _EDITED_DF_KEY,
                _EDITED_FLAG_KEY,
                "dtpi",
            )
            st.stop()

        for w in warnings:
            st.warning(w)

        mapped_df = mapped_training_frame(
            raw_df,
            [col_smiles, col_seq, col_label],
            ["smiles", "sequence", "label"],
        )
        leakage_msgs = leakage_warnings("dtpi", mapped_df)
        if leakage_msgs:
            st.markdown("**Data Leakage Check**")
            for msg in leakage_msgs:
                st.warning(f"Leakage check: {msg}")
        else:
            st.success("Data leakage check: no duplicate-pair or high-overlap risk detected under the random split preview.")

        long_mask = long_sequence_row_mask(raw_df, [col_seq], MAX_DTPI_RESIDUES)
        long_row_count = int(long_mask.sum())
        if long_row_count:
            seq_lens_for_warning = raw_df[col_seq].astype(str).str.strip().str.len()
            max_len_detected = int(seq_lens_for_warning.max())
            st.warning(
                f"{long_row_count:,} pair row(s) contain a protein longer than "
                f"{MAX_DTPI_RESIDUES} residues (max {max_len_detected:,}). Resolve this before training."
            )
            long_preview = raw_df.loc[long_mask, [col_smiles, col_seq, col_label]].head(5).copy()
            long_preview["Protein length"] = seq_lens_for_warning.loc[long_mask].head(5).to_numpy()
            long_preview[col_smiles] = long_preview[col_smiles].astype(str).str[:50] + "…"
            long_preview[col_seq] = long_preview[col_seq].astype(str).str[:50] + "…"
            st.dataframe(long_preview, use_container_width=True, hide_index=True)

            fix_cols = st.columns(2)
            with fix_cols[0]:
                if st.button(
                    f"Trim long proteins to {MAX_DTPI_RESIDUES} residues",
                    key="dtpi_trim_long_sequences",
                    use_container_width=True,
                ):
                    trimmed = trim_sequence_columns(raw_df, [col_seq], MAX_DTPI_RESIDUES)
                    apply_edited_df(trimmed, _EDITED_DF_KEY, _EDITED_FLAG_KEY)
            with fix_cols[1]:
                if st.button(
                    f"Remove {long_row_count:,} long-sequence row(s)",
                    key="dtpi_remove_long_sequence_rows",
                    use_container_width=True,
                ):
                    cleaned = raw_df.loc[~long_mask].copy().reset_index(drop=True)
                    if cleaned.empty:
                        st.error("Removing long-sequence rows would remove all rows. Use trimming or upload a shorter dataset.")
                    else:
                        apply_edited_df(cleaned, _EDITED_DF_KEY, _EDITED_FLAG_KEY)
            st.stop()

        invalid_embedding_mask = invalid_embedding_row_mask(
            raw_df,
            {
                col_smiles: lambda value: (
                    bool(_VALID_SMILES.match(value))
                    and len(value) > 0
                    and not any(ch.isspace() for ch in value)
                ),
                col_seq: lambda value: bool(_VALID_AA.match(value.upper())),
            },
        )
        if render_invalid_embedding_cleanup(
            raw_df,
            invalid_embedding_mask,
            [col_smiles, col_seq, col_label],
            _EDITED_DF_KEY,
            _EDITED_FLAG_KEY,
            "dtpi",
            "compound/protein inputs outside the allowed alphabets or whitespace that can hinder embedding",
        ):
            st.stop()

        seq_lens = raw_df[col_seq].astype(str).str.len()
        mean_seq_len = float(seq_lens.mean())

        _HARD_CAP = 100_000
        if stats["rows"] > _HARD_CAP:
            st.warning(
                f"Dataset has **{stats['rows']:,} pairs** — only up to **{_HARD_CAP:,}** can be used. "
                "Use the positive and negative sliders below to keep the selected total within the limit."
            )

        st.success(
            f"Valid — {stats['rows']:,} pairs · "
            f"{stats['unique_drugs']:,} unique compounds · "
            f"{stats['unique_seqs']:,} unique sequences · "
            f"{stats['n_pos']:,} positive · {stats['n_neg']:,} negative · "
            f"avg seq length {int(mean_seq_len):,} residues"
        )

        preview = raw_df[[col_smiles, col_seq, col_label]].head(5).copy()
        preview.columns = ["smiles (preview)", "sequence (preview)", "label"]
        preview["smiles (preview)"]   = preview["smiles (preview)"].astype(str).str[:50] + "…"
        preview["sequence (preview)"] = preview["sequence (preview)"].astype(str).str[:50] + "…"
        st.dataframe(preview, use_container_width=True, hide_index=True)

        st.markdown("**Data Sampling**")
        sampling_signature = f"{stats['rows']}:{stats['n_pos']}:{stats['n_neg']}:{col_label}"
        if st.session_state.get(_SAMPLING_SIGNATURE_KEY) != sampling_signature:
            st.session_state[_POS_COUNT_KEY] = stats["n_pos"]
            st.session_state[_NEG_COUNT_KEY] = stats["n_neg"]
            st.session_state[_SAMPLING_SIGNATURE_KEY] = sampling_signature

        st.session_state[_POS_COUNT_KEY] = min(
            max(1, int(st.session_state.get(_POS_COUNT_KEY, stats["n_pos"]))),
            stats["n_pos"],
        )
        st.session_state[_NEG_COUNT_KEY] = min(
            max(1, int(st.session_state.get(_NEG_COUNT_KEY, stats["n_neg"]))),
            stats["n_neg"],
        )

        bc1, bc2 = st.columns(2)
        with bc1:
            positive_pairs_use = st.slider(
                "Positive pairs to use",
                min_value=1,
                max_value=stats["n_pos"],
                step=1,
                format="%d",
                help=f"Choose how many label=1 pairs to include. Available: {stats['n_pos']:,}.",
                key=_POS_COUNT_KEY,
            )
        with bc2:
            negative_pairs_use = st.slider(
                "Negative pairs to use",
                min_value=1,
                max_value=stats["n_neg"],
                step=1,
                format="%d",
                help=f"Choose how many label=0 pairs to include. Available: {stats['n_neg']:,}.",
                key=_NEG_COUNT_KEY,
            )

        sc1, sc2 = st.columns(2)
        with sc1:
            n_pairs_use = positive_pairs_use + negative_pairs_use
            st.metric("Selected pairs / limit", f"{n_pairs_use:,} / {_HARD_CAP:,}")
        with sc2:
            train_split = st.slider(
                "Training split (%)",
                min_value=60, max_value=90, value=80, step=5,
                format="%d%%",
                help="Percentage of selected pairs used for training.",
            )

        sample_counts = compute_label_sample_counts(
            positive_pairs_use, negative_pairs_use, stats["n_pos"], stats["n_neg"],
        )
        sampling_over_cap = sample_counts.selected_total > _HARD_CAP
        if sampling_over_cap:
            st.error(
                f"Selected {sample_counts.selected_total:,} pairs, but the DTPI limit is "
                f"{_HARD_CAP:,}. Reduce positive or negative pairs before submitting."
            )
        n_train_pos = int(sample_counts.selected_pos * train_split / 100)
        n_train_neg = int(sample_counts.selected_neg * train_split / 100)
        n_train = n_train_pos + n_train_neg
        n_test = sample_counts.selected_total - n_train
        st.caption(
            f"Selected **{sample_counts.selected_total:,}** pairs "
            f"({sample_counts.selected_pos:,} positive · {sample_counts.selected_neg:,} negative) → "
            f"training **{n_train:,}** ({n_train_pos:,}+{n_train_neg:,}) · "
            f"testing **{n_test:,}**"
        )


    st.session_state[_DATA_CONTEXT_KEY] = {
        "data_ready": data_ready,
        "raw_df": raw_df,
        "col_smiles": col_smiles,
        "col_seq": col_seq,
        "col_label": col_label,
        "mean_seq_len": mean_seq_len,
        "stats": stats,
        "n_pairs_use": n_pairs_use,
        "positive_pairs_use": positive_pairs_use,
        "negative_pairs_use": negative_pairs_use,
        "train_split": train_split,
        "sampling_over_cap": sampling_over_cap,
    }
    _render_step_nav("Data", next_disabled=(not data_ready) or sampling_over_cap)

if active_step == "Architecture":

        # ── 3. Embedding models ───────────────────────────────────────────────────────
    _render_workflow_band("Architecture")
    st.subheader("Embedding Models")

    emb1, emb2 = st.columns(2)

    with emb1:
        st.markdown("**Compound (SMILES)**")
        chem_label = st.selectbox(
            "ChemBERTa model",
            list(CHEMBERTA_OPTIONS.keys()),
            index=0,
            key="dtpi_chem_label",
            help="Encodes SMILES strings into fixed-length compound embeddings.",
        )
        chem_model_name, chem_dim = CHEMBERTA_OPTIONS[chem_label]
        st.caption(f"`{chem_model_name}` · **{chem_dim}**-dim output")

    with emb2:
        st.markdown("**Protein (Sequence)**")
        esm2_label = st.selectbox(
            "ESM2 model",
            list(ESM2_OPTIONS.keys()),
            index=1,
            key="dtpi_esm2_label",
            help="Larger models produce more informative embeddings but require more GPU memory and time.",
        )
        esm_model_name, esm_dim = ESM2_OPTIONS[esm2_label]
        if esm_model_name == "esm2_t33_650M_UR50D":
            st.warning(
                "ESM2 650M is very slow to embed — expect significantly longer run times. "
                "Use 35M or 150M unless you have a large GPU and ample time."
            )
            st.caption(f"`{esm_model_name}` · **{esm_dim}**-dim output")

    embedding_representation = st.radio(
        "Embedding representation",
        ["pooled", "chunked"],
        index=["pooled", "chunked"].index(embedding_representation),
        horizontal=True,
        key="dtpi_embedding_representation",
        help="Pooled stores one vector per side. Chunked stores local window embeddings for both SMILES and protein.",
    )

    smiles_chunk_max_len = 512
    protein_chunk_max_len = MAX_DTPI_RESIDUES
    smiles_num_chunks = 8
    protein_num_chunks = 8
    chunk_dtype = "float16"
    if embedding_representation == "chunked":
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            smiles_num_chunks = st.selectbox("SMILES chunks", [4, 8, 16, 32], index=1, key="dtpi_smiles_chunks")
        with cc2:
            protein_num_chunks = st.selectbox("Protein chunks", [4, 8, 16, 32], index=1, key="dtpi_protein_chunks")
        with cc3:
            chunk_dtype = st.selectbox("Storage precision", ["float16", "float32"], index=0, key="dtpi_chunk_dtype")
        st.caption(
            f"Chunked input: {smiles_num_chunks} SMILES chunks + {protein_num_chunks} protein chunks, "
            f"projected to a shared sequence dimension."
        )

    sequence_mode = embedding_representation == "chunked"
    input_dim = max(chem_dim, esm_dim) if sequence_mode else chem_dim + esm_dim
    st.caption(
        f"Classifier input: **{input_dim}** "
        + (
            f"shared dims per chunk ({chem_dim} ChemBERTa and {esm_dim} ESM2 projected)"
            if sequence_mode
            else f"dims ({chem_dim} compound + {esm_dim} protein, concatenated)"
        )
    )

    st.divider()

    # ── 4. Model Builder ──────────────────────────────────────────────────────────
    st.subheader("Model Builder")

    st.session_state.setdefault("dtpi_layers", default_layers())
    st.session_state.setdefault("_dtpi_lid", 2)

    LAYER_TYPES    = ["linear", "cnn1d", "bilstm", "gru", "transformer", "residual"]
    ACT_OPTIONS    = ["relu", "gelu", "tanh", "elu", "silu", "leaky_relu"]
    KERNEL_OPTIONS = [3, 5, 7, 9]
    NHEAD_OPTIONS  = [2, 4, 8, 16]

    layers: list = st.session_state["dtpi_layers"]

    to_remove = None
    to_move   = None

    for i, layer in enumerate(layers):
        lid = layer["id"]
        with st.container(border=True):
            hdr_cols = st.columns([5, 1, 1, 1])
            with hdr_cols[0]:
                st.markdown(f"**Layer {i + 1} — {layer['type'].upper()}**")
            with hdr_cols[1]:
                if i > 0:
                    if st.button("↑", key=f"dtpi_up_{lid}", help="Move up"):
                        to_move = (i, "up")
            with hdr_cols[2]:
                if i < len(layers) - 1:
                    if st.button("↓", key=f"dtpi_dn_{lid}", help="Move down"):
                        to_move = (i, "down")
            with hdr_cols[3]:
                if st.button("✕", key=f"dtpi_rm_{lid}", help="Remove layer"):
                    to_remove = i

            lt = layer["type"]

            if lt == "linear":
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    layer["hidden_dim"] = st.number_input(
                        "hidden_dim", min_value=32, max_value=2048, step=32,
                        value=int(layer.get("hidden_dim", 256)), key=f"dtpi_hd_{lid}",
                    )
                with c2:
                    layer["activation"] = st.selectbox(
                        "activation", ACT_OPTIONS,
                        index=ACT_OPTIONS.index(layer.get("activation", "relu")),
                        key=f"dtpi_act_{lid}",
                    )
                with c3:
                    layer["dropout"] = st.slider(
                        "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                        key=f"dtpi_drop_{lid}",
                    )
                with c4:
                    layer["batchnorm"] = st.checkbox(
                        "batchnorm", value=bool(layer.get("batchnorm", False)),
                        key=f"dtpi_bn_{lid}",
                    )

            elif lt == "cnn1d":
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    layer["out_channels"] = st.number_input(
                        "out_channels", min_value=8, max_value=512, step=8,
                        value=int(layer.get("out_channels", 64)), key=f"dtpi_och_{lid}",
                    )
                with c2:
                    ks_val = int(layer.get("kernel_size", 3))
                    if ks_val not in KERNEL_OPTIONS:
                        ks_val = 3
                    layer["kernel_size"] = st.selectbox(
                        "kernel_size", KERNEL_OPTIONS,
                        index=KERNEL_OPTIONS.index(ks_val), key=f"dtpi_ks_{lid}",
                    )
                with c3:
                    layer["activation"] = st.selectbox(
                        "activation", ACT_OPTIONS,
                        index=ACT_OPTIONS.index(layer.get("activation", "relu")),
                        key=f"dtpi_act_{lid}",
                    )
                with c4:
                    layer["dropout"] = st.slider(
                        "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                        key=f"dtpi_drop_{lid}",
                    )

            elif lt == "bilstm":
                c1, c2, c3 = st.columns(3)
                with c1:
                    layer["hidden_size"] = st.number_input(
                        "hidden_size", min_value=32, max_value=512, step=32,
                        value=int(layer.get("hidden_size", 128)), key=f"dtpi_hs_{lid}",
                    )
                with c2:
                    layer["num_layers"] = st.number_input(
                        "num_layers", min_value=1, max_value=3,
                        value=int(layer.get("num_layers", 1)), key=f"dtpi_nl_{lid}",
                    )
                with c3:
                    layer["dropout"] = st.slider(
                        "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                        key=f"dtpi_drop_{lid}",
                    )

            elif lt == "gru":
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    layer["hidden_size"] = st.number_input(
                        "hidden_size", min_value=32, max_value=512, step=32,
                        value=int(layer.get("hidden_size", 128)), key=f"dtpi_hs_{lid}",
                    )
                with c2:
                    layer["num_layers"] = st.number_input(
                        "num_layers", min_value=1, max_value=3,
                        value=int(layer.get("num_layers", 1)), key=f"dtpi_nl_{lid}",
                    )
                with c3:
                    layer["bidirectional"] = st.checkbox(
                        "bidirectional", value=bool(layer.get("bidirectional", True)),
                        key=f"dtpi_bidir_{lid}",
                    )
                with c4:
                    layer["dropout"] = st.slider(
                        "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                        key=f"dtpi_drop_{lid}",
                    )

            elif lt == "transformer":
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    d_model_val = int(layer.get("d_model", 256))
                    layer["d_model"] = st.number_input(
                        "d_model", min_value=64, max_value=1024, step=64,
                        value=d_model_val, key=f"dtpi_dm_{lid}",
                    )
                with c2:
                    nhead_val = int(layer.get("nhead", 4))
                    d_now     = int(layer["d_model"])
                    valid_nh  = [h for h in NHEAD_OPTIONS if d_now % h == 0]
                    if not valid_nh:
                        valid_nh = [1]
                    if nhead_val not in valid_nh:
                        nhead_val = valid_nh[0]
                    layer["nhead"] = st.selectbox(
                        "nhead", valid_nh,
                        index=valid_nh.index(nhead_val), key=f"dtpi_nh_{lid}",
                    )
                with c3:
                    layer["num_layers"] = st.number_input(
                        "num_layers", min_value=1, max_value=4,
                        value=int(layer.get("num_layers", 2)), key=f"dtpi_nl_{lid}",
                    )
                with c4:
                    d_now   = int(layer["d_model"])
                    ff_opts = [d_now * 2, d_now * 4]
                    ff_val  = int(layer.get("dim_feedforward", d_now * 2))
                    if ff_val not in ff_opts:
                        ff_val = ff_opts[0]
                    layer["dim_feedforward"] = st.selectbox(
                        "dim_feedforward", ff_opts,
                        index=ff_opts.index(ff_val), key=f"dtpi_ff_{lid}",
                    )
                with c5:
                    layer["dropout"] = st.slider(
                        "dropout", 0.0, 0.5, float(layer.get("dropout", 0.1)), 0.05,
                        key=f"dtpi_drop_{lid}",
                    )

            elif lt == "residual":
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    layer["hidden_dim"] = st.number_input(
                        "hidden_dim", min_value=32, max_value=2048, step=32,
                        value=int(layer.get("hidden_dim", 256)), key=f"dtpi_hd_{lid}",
                    )
                with c2:
                    layer["activation"] = st.selectbox(
                        "activation", ACT_OPTIONS,
                        index=ACT_OPTIONS.index(layer.get("activation", "relu")),
                        key=f"dtpi_act_{lid}",
                    )
                with c3:
                    layer["dropout"] = st.slider(
                        "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                        key=f"dtpi_drop_{lid}",
                    )
                with c4:
                    layer["batchnorm"] = st.checkbox(
                        "batchnorm", value=bool(layer.get("batchnorm", False)),
                        key=f"dtpi_bn_{lid}",
                    )

    # Apply remove / move mutations
    if to_remove is not None:
        layers.pop(to_remove)
        st.session_state["dtpi_layers"] = layers
        st.rerun()

    if to_move is not None:
        idx, direction = to_move
        if direction == "up" and idx > 0:
            layers[idx], layers[idx - 1] = layers[idx - 1], layers[idx]
        elif direction == "down" and idx < len(layers) - 1:
            layers[idx], layers[idx + 1] = layers[idx + 1], layers[idx]
        st.session_state["dtpi_layers"] = layers
        st.rerun()

    # Add layer row
    st.markdown("---")
    add_cols = st.columns([2, 1])
    with add_cols[0]:
        new_type = st.selectbox("Layer type", LAYER_TYPES, key="dtpi_new_layer_type",
                                label_visibility="collapsed")
    with add_cols[1]:
        if st.button("Add Layer", use_container_width=True):
            new_id = st.session_state["_dtpi_lid"]
            st.session_state["_dtpi_lid"] += 1
            defaults_by_type = {
                "linear":      {"type": "linear",      "hidden_dim": 128,  "activation": "relu", "dropout": 0.3, "batchnorm": False},
                "cnn1d":       {"type": "cnn1d",        "out_channels": 64, "kernel_size": 3, "activation": "relu", "dropout": 0.3},
                "bilstm":      {"type": "bilstm",       "hidden_size": 128, "num_layers": 1, "dropout": 0.3},
                "gru":         {"type": "gru",          "hidden_size": 128, "num_layers": 1, "bidirectional": True, "dropout": 0.3},
                "transformer": {"type": "transformer",  "d_model": 256, "nhead": 4, "num_layers": 2, "dim_feedforward": 512, "dropout": 0.1},
                "residual":    {"type": "residual",     "hidden_dim": 256, "activation": "relu", "dropout": 0.3, "batchnorm": False},
            }
            new_layer = {"id": new_id, **defaults_by_type[new_type]}
            layers.append(new_layer)
            st.session_state["dtpi_layers"] = layers
            st.rerun()

    _SEQ_LAYER_TYPES = {"bilstm", "gru", "transformer"}
    if embedding_representation == "pooled" and any(
        l["type"].lower() in _SEQ_LAYER_TYPES for l in layers
    ):
        st.warning(
            "**Pooled embeddings with sequence-aware layers:** BiLSTM, GRU, and Transformer layers "
            "expect a sequence of vectors as input, but **pooled** embeddings collapse each side "
            "to a single vector. These layers will treat it as a 1-step sequence and lose their "
            "advantage. Switch to **chunked** embeddings (above) to get the full benefit of these "
            "architectures, or replace them with **Linear** / **Residual** layers."
        )

    st.divider()

    # ── 5. Architecture visualization ─────────────────────────────────────────────
    layer_configs_display = [
        {k: v for k, v in lyr.items() if k != "id"}
        for lyr in layers
    ]
    n_param = _total_param_count(
        input_dim,
        layer_configs_display,
        sequence_mode=sequence_mode,
        projection_dims=(chem_dim, esm_dim) if sequence_mode else None,
    ) if layers else 0
    param_limit_exceeded = n_param > MAX_MODEL_PARAMS

    st.caption("Architecture preview")
    arch_info_cols = st.columns([1, 2])
    with arch_info_cols[0]:
        if layers:
            st.metric("Approx. parameters", f"{n_param:,}")
            if param_limit_exceeded:
                st.error(f"Maximum allowed: {MAX_MODEL_PARAMS:,} parameters.")
    with arch_info_cols[1]:
        st.caption(
            f"Input dim: **{input_dim}**\n"
            + (
                f"{smiles_num_chunks} ChemBERTa chunks + {protein_num_chunks} ESM2 chunks"
                if sequence_mode
                else f"({chem_dim} ChemBERTa + {esm_dim} ESM2)"
            )
        )

    if layers:
        render_architecture_graph(
            layer_configs_display,
            input_dim=input_dim,
            input_label="Input",
            input_subtitle=(
                f"{smiles_num_chunks} ChemBERTa chunks + {protein_num_chunks} ESM2 chunks"
                if sequence_mode
                else f"{chem_dim} ChemBERTa + {esm_dim} ESM2"
            ),
            key="dtpi_architecture_graph",
        )
    else:
        st.info("Add at least one layer to preview the architecture.")

    _render_step_nav("Architecture")

if active_step == "Training":

        # ── 6. Training hyperparameters ───────────────────────────────────────────────
    _render_workflow_band("Training")
    st.subheader("Training Parameters")

    tp1, tp2, tp3, tp4 = st.columns(4)
    with tp1:
        epochs = st.slider("Epochs", min_value=5, max_value=100, value=30, step=5)
    with tp2:
        lr = st.selectbox(
            "Learning rate", [0.001, 0.0005, 0.0001],
            format_func=lambda x: f"{x:.4f}",
        )
    with tp3:
        batch_size = st.selectbox("Batch size", [32, 64, 128], index=1)
    with tp4:
        early_stop = st.selectbox(
            "Early stopping patience",
            [0, 5, 10, 15],
            format_func=lambda x: "Disabled" if x == 0 else f"{x} epochs without improvement",
            index=1,
        )

    est = _estimate_time(
        n_pairs_use, stats["unique_seqs"], mean_seq_len,
        stats["unique_drugs"], epochs, batch_size,
    )
    st.caption(
        f"**Estimated time:** {est} _(rough, GPU-dependent)_   |   "
        f"All jobs auto-stopped after **4 hours**."
    )

    st.divider()

    # ── 7. Submit ─────────────────────────────────────────────────────────────────
    if not layers:
        st.error("Add at least one layer before submitting.")
        st.stop()

    notify_email = st.text_input(
        "Notify me by email when done (optional)",
        placeholder="your@email.com",
        key="dtpi_notify_email",
    )

    submit_disabled = (not data_ready) or param_limit_exceeded or sampling_over_cap
    if st.button("Submit Training Job", type="primary", use_container_width=True, disabled=submit_disabled):
        if sampling_over_cap:
            st.error("Reduce selected positive or negative pairs to 100,000 total before submitting.")
            st.stop()
        if param_limit_exceeded:
            st.error(f"Model has {n_param:,} parameters; reduce it to {MAX_MODEL_PARAMS:,} or fewer before submitting.")
            st.stop()
        hp = {
            "task_type":           "dtpi",
            "chem_model":          chem_model_name,
            "chem_dim":            chem_dim,
            "esm_model":           esm_model_name,
            "esm_dim":             esm_dim,
            "embedding_representation": embedding_representation,
            "smiles_chunk_max_len": smiles_chunk_max_len,
            "smiles_num_chunks":    smiles_num_chunks,
            "protein_chunk_max_len": protein_chunk_max_len,
            "protein_num_chunks":   protein_num_chunks,
            "chunk_model_dim":      input_dim,
            "chunk_dtype":          chunk_dtype,
            "input_dim":           input_dim,
            "layer_configs":       layer_configs_display,
            "epochs":              epochs,
            "learning_rate":       lr,
            "batch_size":          batch_size,
            "early_stop_patience": early_stop,
            "train_split":         train_split / 100,
            "notify_email":        notify_email.strip(),
        }
        assert raw_df is not None
        send_df = raw_df[[col_smiles, col_seq, col_label]].copy()
        send_df.columns = ["smiles", "sequence", "label"]
        send_df["label"] = send_df["label"].astype(float).astype(int)
        send_df, _ = sample_by_label_counts(
            send_df, "label", positive_pairs_use, negative_pairs_use, random_state=42,
        )
        csv_bytes = send_df.to_csv(index=False).encode()

        with st.spinner("Submitting..."):
            try:
                r = requests.post(
                    f"{BACKEND}/create_job",
                    files=[("files", ("training_data.csv", csv_bytes, "text/csv"))],
                    data={"hyperparams": json.dumps(hp)},
                )
                r.raise_for_status()
                data = r.json()

                st.session_state["last_run_id"]       = data["run_id"]
                st.session_state["last_cancel_token"]  = data["cancel_token"]

                st.success(f"Job submitted — Run ID: `{data['run_id']}`")
                st.warning("**Save your cancel token — it will not be shown again.**")
                st.code(data["cancel_token"], language=None)
                st.info("Go to **Tools → Check Results** to monitor training progress.")
            except Exception as e:
                st.error(f"Submission failed: {e}")

    st.divider()
    _render_step_nav("Training")
