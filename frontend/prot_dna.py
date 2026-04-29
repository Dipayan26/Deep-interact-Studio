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

from data_sampling import balanced_sample_by_label, compute_balanced_sample_counts
from validation_recovery import (
    build_recoverable_row_mask,
    clear_edited_df,
    render_edited_download,
    render_recovery_controls,
)

BACKEND = os.getenv("BACKEND_URL", "http://backend:8005")
MAX_MODEL_PARAMS = 10_000_000

_VALID_AA  = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBJOUXZ*\-]+$")
_VALID_DNA = re.compile(r"^[ATGCNatgcn]+$")

# ── Embedding model options ───────────────────────────────────────────────────
DNABERT_OPTIONS = {
    "DNABERT 6-mer (768-dim, default)": ("armheb/DNA_bert_6", 768),
}

ESM2_OPTIONS = {
    "ESM2 8M  (320-dim, fastest)":   ("esm2_t6_8M_UR50D",    320),
    "ESM2 35M (480-dim, default)":   ("esm2_t12_35M_UR50D",  480),
    "ESM2 150M (640-dim, accurate)": ("esm2_t30_150M_UR50D", 640),
    "ESM2 650M (1280-dim, slow)":    ("esm2_t33_650M_UR50D", 1280),
}

# ── Demo data generator ───────────────────────────────────────────────────────

def _generate_demo_csv() -> str:
    """Generate synthetic PDI demo data (50 pairs) with a fixed random seed."""
    rng = random.Random(99)

    DNA_BASES   = list("ATGC")
    DNA_WEIGHTS = [0.25, 0.25, 0.25, 0.25]

    AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
    AA_WEIGHTS  = [
        0.0669, 0.0166, 0.0545, 0.0672, 0.0386,
        0.0709, 0.0228, 0.0580, 0.0589, 0.0966,
        0.0242, 0.0406, 0.0472, 0.0393, 0.0553,
        0.0664, 0.0534, 0.0686, 0.0109, 0.0292,
    ]

    # Classic TF-binding motifs used in positive pairs
    TF_MOTIFS = [
        "TATAAA",    # TATA-box
        "CCGCGG",    # SP1 GC-box
        "CACGTG",    # E-box (bHLH)
        "TGACTCA",   # AP-1 CRE
        "AGGAGG",    # Shine-Dalgarno-like
        "GGGCGG",    # GC-rich SP1
        "ATGCCAT",   # generic initiator
    ]

    def _rand_dna(length: int, with_motif: bool) -> str:
        seq = "".join(rng.choices(DNA_BASES, weights=DNA_WEIGHTS, k=length))
        if with_motif and rng.random() < 0.75:
            motif = rng.choice(TF_MOTIFS)
            pos   = rng.randint(0, max(0, length - len(motif)))
            seq   = seq[:pos] + motif + seq[pos + len(motif):]
        return seq[:length]

    def _rand_prot(length: int) -> str:
        return "".join(rng.choices(AA_ALPHABET, weights=AA_WEIGHTS, k=length))

    rows = []
    used: set = set()
    for label in ([1] * 25 + [0] * 25):
        for _ in range(300):
            dna  = _rand_dna(rng.randint(30, 100), with_motif=(label == 1))
            prot = _rand_prot(rng.randint(60, 150))
            key  = (dna[:20], prot[:20])
            if key not in used:
                used.add(key)
                rows.append({"dna_sequence": dna, "protein_sequence": prot, "label": label})
                break

    rng.shuffle(rows)
    lines = ["dna_sequence,protein_sequence,label"]
    for r in rows:
        lines.append(f"{r['dna_sequence']},{r['protein_sequence']},{r['label']}")
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


def _total_param_count(input_dim: int, layer_configs: list) -> int:
    total = 0
    cur   = input_dim
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
            total += 1 * out_ch * k + out_ch
            cur = out_ch
        elif lt == "bilstm":
            h    = int(cfg.get("hidden_size", 128))
            nl   = int(cfg.get("num_layers", 1))
            gate = 4
            total += gate * (cur * h + h * h + h)
            total += gate * (cur * h + h * h + h)
            for _ in range(nl - 1):
                total += 2 * gate * (2 * h * h + h)
            cur = 2 * h
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
    total += cur * 1 + 1
    return total


# ── Architecture figure ───────────────────────────────────────────────────────

def _arch_figure(layer_configs: list, dna_dim: int, prot_dim: int) -> plt.Figure:
    BG        = "#FFF8F0"   # warm amber tint
    input_dim = dna_dim + prot_dim

    entries = [{
        "name": "Input",
        "dim":  input_dim,
        "sub":  f"concat(DNA,prot)\n{input_dim}-dim",
    }]
    cur_dim = input_dim
    for i, cfg in enumerate(layer_configs):
        lt      = cfg["type"]
        out_dim = _compute_out_dim(lt, cur_dim, cfg)
        entries.append({"name": f"Layer {i + 1}", "dim": out_dim, "sub": lt.upper()})
        cur_dim = out_dim
    entries.append({"name": "Output", "dim": 1, "sub": "sigmoid(logit)"})

    n    = len(entries)
    COLS = ["#C06000"] + ["#A05000"] * (n - 2) + ["#7A3800"]

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

def _validate(df, col_dna, col_prot, col_label):
    errors, warnings = [], []

    for col in [col_dna, col_prot, col_label]:
        n = df[col].isnull().sum()
        if n:
            errors.append(f"Column **{col}** has {n} missing value(s).")
    if errors:
        return errors, warnings, {}

    # DNA: only A/T/G/C/N
    sample_dna = df[col_dna].head(200)
    bad_dna = sample_dna.astype(str).str.strip().str.upper().apply(
        lambda s: not bool(_VALID_DNA.match(s)) or len(s) == 0
    )
    if bad_dna.sum():
        first = sample_dna[bad_dna].iloc[0][:30]
        errors.append(
            f"Column **{col_dna}** contains {int(bad_dna.sum())} sequence(s) with invalid "
            f"characters (first: `{first}...`). Only A, T, G, C (and N) accepted."
        )

    # Protein: amino acid alphabet
    sample_aa = df[col_prot].head(200)
    bad_aa = sample_aa.astype(str).str.strip().str.upper().apply(
        lambda s: not bool(_VALID_AA.match(s)) or len(s) == 0
    )
    if bad_aa.sum():
        first = sample_aa[bad_aa].iloc[0][:30]
        errors.append(
            f"Column **{col_prot}** contains {int(bad_aa.sum())} sequence(s) with invalid "
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
        "rows":         n_rows,
        "unique_dna":   df[col_dna].nunique(),
        "unique_prots": df[col_prot].nunique(),
        "n_pos": int((df[col_label].astype(float).astype(int) == 1).sum()) if not errors else 0,
        "n_neg": int((df[col_label].astype(float).astype(int) == 0).sum()) if not errors else 0,
    }
    return errors, warnings, stats


def _estimate_time(n_pairs: int, n_unique_dna: int, mean_dna_len: float,
                   n_unique_prots: int, mean_prot_len: float,
                   epochs: int, batch_size: int) -> str:
    dna_embed_s  = (n_unique_dna   / 16) * 0.20 * max(1.0, mean_dna_len  / 512)
    prot_embed_s = (n_unique_prots /  8) * 0.25 * max(1.0, mean_prot_len / 512)
    train_s      = epochs * math.ceil(n_pairs * 0.8 / batch_size) * 0.04
    total        = dna_embed_s + prot_embed_s + train_s
    if total < 90:
        return f"~{int(total)} s"
    if total < 3600:
        return f"~{int(total / 60)} min"
    return f"~{total / 3600:.1f} h"


# =============================================================================
# Page
# =============================================================================

# ── PDI amber theme ───────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] > section:first-child::before {
    content: "";
    display: block;
    height: 4px;
    background: linear-gradient(90deg, #C06000, #E08020);
    margin-bottom: 0;
}
h2, h3 { color: #8B4500 !important; }
div[data-testid="stButton"] button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background-color: #C06000 !important;
    border-color: #C06000 !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background-color: #A05000 !important;
    border-color: #A05000 !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #C06000 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Protein–DNA Interaction")
st.caption("Train a classifier to predict binding between DNA-binding proteins and DNA sequences.")
st.divider()

# ── 1. Training data ──────────────────────────────────────────────────────────
st.subheader("Training Data")

st.session_state.setdefault("pdi_demo_loaded",  False)
st.session_state.setdefault("pdi_uploader_key", 0)
_EDITED_DF_KEY = "pdi_edited_df"
_EDITED_FLAG_KEY = "pdi_data_edited"
_SOURCE_KEY = "pdi_data_source"
st.session_state.setdefault(_EDITED_FLAG_KEY, False)

col_dl, col_ex, _, col_reset = st.columns([2, 2, 2, 1])
with col_dl:
    st.download_button(
        "Download demo CSV", data=_DEMO_CSV,
        file_name="demo_pdi_train.csv", mime="text/csv",
        help="Download the synthetic demo CSV for offline inspection.",
        use_container_width=True,
    )
with col_ex:
    if st.button("Load Example Data",
                 help="Loads 50 synthetic protein-DNA pairs to test the workflow.",
                 use_container_width=True):
        clear_edited_df(_EDITED_DF_KEY, _EDITED_FLAG_KEY)
        st.session_state["pdi_demo_loaded"] = True
        st.session_state["pdi_uploader_key"] += 1
        st.rerun()
with col_reset:
    if st.button("Reset", help="Clear loaded data and start fresh.", use_container_width=True):
        clear_edited_df(_EDITED_DF_KEY, _EDITED_FLAG_KEY)
        st.session_state["pdi_demo_loaded"] = False
        st.session_state["pdi_uploader_key"] += 1
        st.rerun()

st.caption(
    "Upload a CSV with any column names. "
    "You will map them to **DNA Sequence**, **Protein Sequence**, and **Label (0/1)** below."
)

uploaded = st.file_uploader(
    "Upload CSV", type=["csv"], label_visibility="collapsed",
    key=f"pdi_uploader_{st.session_state['pdi_uploader_key']}",
)

demo_loaded = st.session_state.get("pdi_demo_loaded", False)
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
    st.info("Using example data (50 synthetic protein-DNA pairs). Upload your own CSV to override.")
    raw_df     = pd.read_csv(io.StringIO(_DEMO_CSV))
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
    render_edited_download(raw_df, _EDITED_FLAG_KEY, "edited_pdi_train.csv")

if not data_ready:
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] > div:has(.pdi-grey-marker) ~ div {
        opacity: 0.4;
        pointer-events: none;
        user-select: none;
    }
    </style>
    <span class="pdi-grey-marker"></span>
    """, unsafe_allow_html=True)

st.divider()

# ── 2. Column mapping ─────────────────────────────────────────────────────────
st.subheader("Column Mapping")

_placeholder_cols = ["dna_sequence", "protein_sequence", "label"]
_sel_cols = raw_df.columns.tolist() if data_ready else _placeholder_cols

mc1, mc2, mc3 = st.columns(3)
with mc1:
    col_dna   = st.selectbox("DNA Sequence",     _sel_cols, index=0)
with mc2:
    col_prot  = st.selectbox("Protein Sequence", _sel_cols, index=min(1, len(_sel_cols) - 1))
with mc3:
    col_label = st.selectbox("Label (0/1)",      _sel_cols, index=min(2, len(_sel_cols) - 1))

mean_dna_len  = 0.0
mean_prot_len = 0.0
stats         = {"rows": 0, "unique_dna": 0, "unique_prots": 0, "n_pos": 0, "n_neg": 0}
n_pairs_use   = 0
train_split   = 80
pos_class_percent = 50
neg_class_percent = 50

if data_ready:
    if col_dna == col_prot:
        st.error("DNA Sequence and Protein Sequence cannot be the same column.")
        st.stop()

    errors, warnings, stats = _validate(raw_df, col_dna, col_prot, col_label)
    affected_mask = build_recoverable_row_mask(
        raw_df,
        col_dna,
        col_prot,
        col_label,
        lambda value: bool(_VALID_DNA.match(value.upper())),
        lambda value: bool(_VALID_AA.match(value.upper())),
    )

    if errors:
        for e in errors:
            st.error(e)
        render_recovery_controls(
            raw_df,
            affected_mask,
            col_dna,
            col_prot,
            col_label,
            _EDITED_DF_KEY,
            _EDITED_FLAG_KEY,
            "pdi",
        )
        st.stop()

    for w in warnings:
        st.warning(w)

    dna_lens      = raw_df[col_dna].astype(str).str.len()
    prot_lens     = raw_df[col_prot].astype(str).str.len()
    mean_dna_len  = float(dna_lens.mean())
    mean_prot_len = float(prot_lens.mean())
    n_long_prot   = int((prot_lens > 1022).sum())

    if n_long_prot > 0:
        st.info(f"{n_long_prot:,} protein sequence(s) exceed 1022 residues — sliding-window embedding will be used.")

    _HARD_CAP = 3000
    if stats["rows"] > _HARD_CAP:
        st.warning(f"Dataset has **{stats['rows']:,} pairs** — only up to **{_HARD_CAP:,}** can be used.")

    st.success(
        f"Valid — {stats['rows']:,} pairs · "
        f"{stats['unique_dna']:,} unique DNA sequences · "
        f"{stats['unique_prots']:,} unique proteins · "
        f"{stats['n_pos']:,} positive · {stats['n_neg']:,} negative · "
        f"avg DNA length {int(mean_dna_len):,} bp · "
        f"avg protein length {int(mean_prot_len):,} aa"
    )

    preview = raw_df[[col_dna, col_prot, col_label]].head(5).copy()
    preview.columns = ["dna_sequence (preview)", "protein_sequence (preview)", "label"]
    preview["dna_sequence (preview)"]     = preview["dna_sequence (preview)"].astype(str).str[:50] + "…"
    preview["protein_sequence (preview)"] = preview["protein_sequence (preview)"].astype(str).str[:50] + "…"
    st.dataframe(preview, use_container_width=True, hide_index=True)

    _max_pairs = min(stats["rows"], _HARD_CAP)
    st.markdown("**Data Sampling**")
    sc1, sc2 = st.columns(2)
    with sc1:
        n_pairs_use = st.slider(
            "Pairs to use",
            min_value=min(20, _max_pairs),
            max_value=_max_pairs,
            value=_max_pairs,
            step=10,
            help=f"Choose how many of your {stats['rows']:,} pairs to use. Maximum is {_HARD_CAP:,}.",
        )
    with sc2:
        train_split = st.slider(
            "Training split (%)",
            min_value=60, max_value=90, value=80, step=5,
            format="%d%%",
            help="Percentage of selected pairs used for training.",
        )

    _POS_BALANCE_KEY = "pdi_positive_percent"
    _NEG_BALANCE_KEY = "pdi_negative_percent"
    st.session_state.setdefault(_POS_BALANCE_KEY, 50)
    st.session_state.setdefault(_NEG_BALANCE_KEY, 50)

    def _sync_pdi_negative_percent():
        st.session_state[_NEG_BALANCE_KEY] = 100 - int(st.session_state[_POS_BALANCE_KEY])

    def _sync_pdi_positive_percent():
        st.session_state[_POS_BALANCE_KEY] = 100 - int(st.session_state[_NEG_BALANCE_KEY])

    bc1, bc2 = st.columns(2)
    with bc1:
        st.slider(
            "Positive pairs (%)",
            min_value=5, max_value=95, step=5,
            key=_POS_BALANCE_KEY,
            format="%d%%",
            on_change=_sync_pdi_negative_percent,
            help="Percentage of selected pairs sampled from label=1 rows.",
        )
    with bc2:
        st.slider(
            "Negative pairs (%)",
            min_value=5, max_value=95, step=5,
            key=_NEG_BALANCE_KEY,
            format="%d%%",
            on_change=_sync_pdi_positive_percent,
            help="Percentage of selected pairs sampled from label=0 rows.",
        )

    pos_class_percent = int(st.session_state[_POS_BALANCE_KEY])
    neg_class_percent = int(st.session_state[_NEG_BALANCE_KEY])

    sample_counts = compute_balanced_sample_counts(
        n_pairs_use, stats["n_pos"], stats["n_neg"], pos_class_percent,
    )
    if sample_counts.selected_total < sample_counts.requested_total:
        st.warning(
            f"Requested {sample_counts.requested_total:,} pairs, but only "
            f"{sample_counts.selected_total:,} can be selected with the current class balance "
            "because one class has fewer available rows."
        )
    n_train_pos = int(sample_counts.selected_pos * train_split / 100)
    n_train_neg = int(sample_counts.selected_neg * train_split / 100)
    n_train = n_train_pos + n_train_neg
    n_test = sample_counts.selected_total - n_train
    st.caption(
        f"Selected **{sample_counts.selected_total:,}** pairs "
        f"({sample_counts.selected_pos:,} positive · {sample_counts.selected_neg:,} negative; "
        f"{pos_class_percent}%/{neg_class_percent}%) → "
        f"training **{n_train:,}** ({n_train_pos:,}+{n_train_neg:,}) · "
        f"testing **{n_test:,}**"
    )

st.divider()

# ── 3. Embedding models ───────────────────────────────────────────────────────
st.subheader("Embedding Models")

emb1, emb2 = st.columns(2)

with emb1:
    st.markdown("**DNA Sequence**")
    dna_label = st.selectbox(
        "DNABERT 6-mer model",
        list(DNABERT_OPTIONS.keys()),
        index=0,
        help="Encodes DNA sequences into fixed-length embeddings using DNABERT-2.",
    )
    dna_model_name, dna_dim = DNABERT_OPTIONS[dna_label]
    st.caption(f"`{dna_model_name}` · **{dna_dim}**-dim output")

with emb2:
    st.markdown("**Protein Sequence**")
    esm2_label = st.selectbox(
        "ESM2 model",
        list(ESM2_OPTIONS.keys()),
        index=1,
        help="Larger models produce more informative embeddings but require more GPU memory and time.",
    )
    esm_model_name, esm_dim = ESM2_OPTIONS[esm2_label]
    if esm_model_name == "esm2_t33_650M_UR50D":
        st.warning(
            "ESM2 650M is very slow to embed — expect significantly longer run times. "
            "Use 35M or 150M unless you have a large GPU and ample time."
        )
    st.caption(f"`{esm_model_name}` · **{esm_dim}**-dim output")

input_dim = dna_dim + esm_dim
st.caption(
    f"Classifier input: **{input_dim}** dims "
    f"({dna_dim} DNABERT-2 + {esm_dim} ESM2, concatenated)"
)

st.divider()

# ── 4. Model Builder ──────────────────────────────────────────────────────────
st.subheader("Model Builder")

st.session_state.setdefault("pdi_layers", [
    {"id": 0, "type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3, "batchnorm": False},
    {"id": 1, "type": "linear", "hidden_dim": 64,  "activation": "relu", "dropout": 0.2, "batchnorm": False},
])
st.session_state.setdefault("_pdi_lid", 2)

LAYER_TYPES    = ["linear", "cnn1d", "bilstm", "gru", "transformer", "residual"]
ACT_OPTIONS    = ["relu", "gelu", "tanh", "elu", "silu", "leaky_relu"]
KERNEL_OPTIONS = [3, 5, 7, 9]
NHEAD_OPTIONS  = [2, 4, 8, 16]

layers: list = st.session_state["pdi_layers"]

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
                if st.button("↑", key=f"pdi_up_{lid}", help="Move up"):
                    to_move = (i, "up")
        with hdr_cols[2]:
            if i < len(layers) - 1:
                if st.button("↓", key=f"pdi_dn_{lid}", help="Move down"):
                    to_move = (i, "down")
        with hdr_cols[3]:
            if st.button("✕", key=f"pdi_rm_{lid}", help="Remove layer"):
                to_remove = i

        lt = layer["type"]

        if lt == "linear":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                layer["hidden_dim"] = st.number_input(
                    "hidden_dim", min_value=32, max_value=2048, step=32,
                    value=int(layer.get("hidden_dim", 256)), key=f"pdi_hd_{lid}",
                )
            with c2:
                layer["activation"] = st.selectbox(
                    "activation", ACT_OPTIONS,
                    index=ACT_OPTIONS.index(layer.get("activation", "relu")),
                    key=f"pdi_act_{lid}",
                )
            with c3:
                layer["dropout"] = st.slider(
                    "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                    key=f"pdi_drop_{lid}",
                )
            with c4:
                layer["batchnorm"] = st.checkbox(
                    "batchnorm", value=bool(layer.get("batchnorm", False)),
                    key=f"pdi_bn_{lid}",
                )

        elif lt == "cnn1d":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                layer["out_channels"] = st.number_input(
                    "out_channels", min_value=8, max_value=512, step=8,
                    value=int(layer.get("out_channels", 64)), key=f"pdi_och_{lid}",
                )
            with c2:
                ks_val = int(layer.get("kernel_size", 3))
                if ks_val not in KERNEL_OPTIONS:
                    ks_val = 3
                layer["kernel_size"] = st.selectbox(
                    "kernel_size", KERNEL_OPTIONS,
                    index=KERNEL_OPTIONS.index(ks_val), key=f"pdi_ks_{lid}",
                )
            with c3:
                layer["activation"] = st.selectbox(
                    "activation", ACT_OPTIONS,
                    index=ACT_OPTIONS.index(layer.get("activation", "relu")),
                    key=f"pdi_act_{lid}",
                )
            with c4:
                layer["dropout"] = st.slider(
                    "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                    key=f"pdi_drop_{lid}",
                )

        elif lt == "bilstm":
            c1, c2, c3 = st.columns(3)
            with c1:
                layer["hidden_size"] = st.number_input(
                    "hidden_size", min_value=32, max_value=512, step=32,
                    value=int(layer.get("hidden_size", 128)), key=f"pdi_hs_{lid}",
                )
            with c2:
                layer["num_layers"] = st.number_input(
                    "num_layers", min_value=1, max_value=3,
                    value=int(layer.get("num_layers", 1)), key=f"pdi_nl_{lid}",
                )
            with c3:
                layer["dropout"] = st.slider(
                    "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                    key=f"pdi_drop_{lid}",
                )

        elif lt == "gru":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                layer["hidden_size"] = st.number_input(
                    "hidden_size", min_value=32, max_value=512, step=32,
                    value=int(layer.get("hidden_size", 128)), key=f"pdi_hs_{lid}",
                )
            with c2:
                layer["num_layers"] = st.number_input(
                    "num_layers", min_value=1, max_value=3,
                    value=int(layer.get("num_layers", 1)), key=f"pdi_nl_{lid}",
                )
            with c3:
                layer["bidirectional"] = st.checkbox(
                    "bidirectional", value=bool(layer.get("bidirectional", True)),
                    key=f"pdi_bidir_{lid}",
                )
            with c4:
                layer["dropout"] = st.slider(
                    "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                    key=f"pdi_drop_{lid}",
                )

        elif lt == "transformer":
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                d_model_val = int(layer.get("d_model", 256))
                layer["d_model"] = st.number_input(
                    "d_model", min_value=64, max_value=1024, step=64,
                    value=d_model_val, key=f"pdi_dm_{lid}",
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
                    index=valid_nh.index(nhead_val), key=f"pdi_nh_{lid}",
                )
            with c3:
                layer["num_layers"] = st.number_input(
                    "num_layers", min_value=1, max_value=4,
                    value=int(layer.get("num_layers", 2)), key=f"pdi_nl_{lid}",
                )
            with c4:
                d_now   = int(layer["d_model"])
                ff_opts = [d_now * 2, d_now * 4]
                ff_val  = int(layer.get("dim_feedforward", d_now * 2))
                if ff_val not in ff_opts:
                    ff_val = ff_opts[0]
                layer["dim_feedforward"] = st.selectbox(
                    "dim_feedforward", ff_opts,
                    index=ff_opts.index(ff_val), key=f"pdi_ff_{lid}",
                )
            with c5:
                layer["dropout"] = st.slider(
                    "dropout", 0.0, 0.5, float(layer.get("dropout", 0.1)), 0.05,
                    key=f"pdi_drop_{lid}",
                )

        elif lt == "residual":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                layer["hidden_dim"] = st.number_input(
                    "hidden_dim", min_value=32, max_value=2048, step=32,
                    value=int(layer.get("hidden_dim", 256)), key=f"pdi_hd_{lid}",
                )
            with c2:
                layer["activation"] = st.selectbox(
                    "activation", ACT_OPTIONS,
                    index=ACT_OPTIONS.index(layer.get("activation", "relu")),
                    key=f"pdi_act_{lid}",
                )
            with c3:
                layer["dropout"] = st.slider(
                    "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                    key=f"pdi_drop_{lid}",
                )
            with c4:
                layer["batchnorm"] = st.checkbox(
                    "batchnorm", value=bool(layer.get("batchnorm", False)),
                    key=f"pdi_bn_{lid}",
                )

# Apply remove / move mutations
if to_remove is not None:
    layers.pop(to_remove)
    st.session_state["pdi_layers"] = layers
    st.rerun()

if to_move is not None:
    idx, direction = to_move
    if direction == "up" and idx > 0:
        layers[idx], layers[idx - 1] = layers[idx - 1], layers[idx]
    elif direction == "down" and idx < len(layers) - 1:
        layers[idx], layers[idx + 1] = layers[idx + 1], layers[idx]
    st.session_state["pdi_layers"] = layers
    st.rerun()

# Add layer row
st.markdown("---")
add_cols = st.columns([2, 1])
with add_cols[0]:
    new_type = st.selectbox("Layer type", LAYER_TYPES, key="pdi_new_layer_type",
                            label_visibility="collapsed")
with add_cols[1]:
    if st.button("Add Layer", use_container_width=True):
        new_id = st.session_state["_pdi_lid"]
        st.session_state["_pdi_lid"] += 1
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
        st.session_state["pdi_layers"] = layers
        st.rerun()

st.divider()

# ── 5. Architecture visualization ─────────────────────────────────────────────
layer_configs_display = [
    {k: v for k, v in lyr.items() if k != "id"}
    for lyr in layers
]
n_param = _total_param_count(input_dim, layer_configs_display) if layers else 0
param_limit_exceeded = n_param > MAX_MODEL_PARAMS

col_viz, col_info = st.columns([3, 1])
with col_viz:
    st.caption("Architecture preview")
    if layers:
        fig_arch = _arch_figure(layer_configs_display, dna_dim, esm_dim)
        st.pyplot(fig_arch, use_container_width=True)
        plt.close(fig_arch)
    else:
        st.info("Add at least one layer to preview the architecture.")

with col_info:
    if layers:
        st.metric("Approx. parameters", f"{n_param:,}")
        if param_limit_exceeded:
            st.error(f"Maximum allowed: {MAX_MODEL_PARAMS:,} parameters.")
    st.caption(
        f"Input dim: **{input_dim}**\n"
        f"({dna_dim} DNABERT-2 + {esm_dim} ESM2)"
    )

st.divider()

# ── 6. Training hyperparameters ───────────────────────────────────────────────
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
    stats["rows"], stats["unique_dna"], mean_dna_len,
    stats["unique_prots"], mean_prot_len, epochs, batch_size,
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
    key="pdi_notify_email",
)

st.warning(
    "Data leakage risk: avoid duplicate pairs and heavy overlap of DNA/protein entities between train and validation. "
    "Prefer entity-disjoint splits for stricter evaluation."
)

submit_disabled = (not data_ready) or param_limit_exceeded
if st.button("Submit Training Job", type="primary", use_container_width=True, disabled=submit_disabled):
    if param_limit_exceeded:
        st.error(f"Model has {n_param:,} parameters; reduce it to {MAX_MODEL_PARAMS:,} or fewer before submitting.")
        st.stop()
    hp = {
        "task_type":           "pdi",
        "dna_model":           dna_model_name,
        "dna_dim":             dna_dim,
        "esm_model":           esm_model_name,
        "esm_dim":             esm_dim,
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
    send_df = raw_df[[col_dna, col_prot, col_label]].copy()
    send_df.columns = ["dna_sequence", "protein_sequence", "label"]
    send_df["label"]        = send_df["label"].astype(float).astype(int)
    send_df["dna_sequence"] = send_df["dna_sequence"].astype(str).str.strip().str.upper()
    send_df, _ = balanced_sample_by_label(
        send_df, "label", n_pairs_use, pos_class_percent, random_state=42,
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

            st.session_state["last_run_id"]      = data["run_id"]
            st.session_state["last_cancel_token"] = data["cancel_token"]

            st.success(f"Job submitted — Run ID: `{data['run_id']}`")
            for msg in data.get("leakage_warnings", []):
                st.warning(f"Leakage check: {msg}")
            st.warning("**Save your cancel token — it will not be shown again.**")
            st.code(data["cancel_token"], language=None)
            st.info("Go to **Tools → Check Results** to monitor training progress.")
        except Exception as e:
            st.error(f"Submission failed: {e}")
