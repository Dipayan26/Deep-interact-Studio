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

BACKEND = os.getenv("BACKEND_URL", "http://backend:8005")

_VALID_AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBJOUXZ*\-]+$")

# ── ESM2 model options ────────────────────────────────────────────────────────
ESM2_OPTIONS = {
    "ESM2 8M  (320-dim, fastest)":   ("esm2_t6_8M_UR50D",    320),
    "ESM2 35M (480-dim, default)":   ("esm2_t12_35M_UR50D",  480),
    "ESM2 150M (640-dim, accurate)": ("esm2_t30_150M_UR50D", 640),
    "ESM2 650M (1280-dim, slow)":    ("esm2_t33_650M_UR50D", 1280),
}

# ── Demo data generator ───────────────────────────────────────────────────────

def _generate_demo_csv() -> str:
    """Generate synthetic demo PPI data using a fixed random seed."""
    rng = random.Random(42)

    AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
    # Realistic amino acid composition weights (roughly based on Swiss-Prot frequencies)
    AA_WEIGHTS = [
        0.0669, 0.0166, 0.0545, 0.0672, 0.0386,
        0.0709, 0.0228, 0.0580, 0.0589, 0.0966,
        0.0242, 0.0406, 0.0472, 0.0393, 0.0553,
        0.0664, 0.0534, 0.0686, 0.0109, 0.0292,
    ]

    def _rand_seq(length: int) -> str:
        return "".join(rng.choices(AA_ALPHABET, weights=AA_WEIGHTS, k=length))

    # Generate 60 unique synthetic sequences (80-180 AA each)
    sequences = []
    seen = set()
    while len(sequences) < 60:
        length = rng.randint(80, 180)
        seq = _rand_seq(length)
        if seq not in seen:
            seen.add(seq)
            sequences.append(seq)

    # Build 100 pairs: 50 positive, 50 negative
    rows = []
    used_pairs = set()

    def _add_pair(label: int) -> bool:
        for _ in range(200):
            a, b = rng.sample(sequences, 2)
            key  = (min(a, b), max(a, b))
            if key not in used_pairs:
                used_pairs.add(key)
                rows.append({"proteinA": a, "proteinB": b, "label": label})
                return True
        return False

    for _ in range(50):
        _add_pair(1)
    for _ in range(50):
        _add_pair(0)

    rng.shuffle(rows)

    lines = ["proteinA,proteinB,label"]
    for r in rows:
        lines.append(f"{r['proteinA']},{r['proteinB']},{r['label']}")
    return "\n".join(lines) + "\n"


_DEMO_CSV = _generate_demo_csv()

# ── Architecture dimension calculator ────────────────────────────────────────

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
        return in_dim  # preserves dimension
    return in_dim


def _total_param_count(input_dim: int, layer_configs: list) -> int:
    """Rough parameter count estimate."""
    total = 0
    cur = input_dim
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
            total += 1 * out_ch * k + out_ch  # in_channels=1
            cur = out_ch
        elif lt == "bilstm":
            h    = int(cfg.get("hidden_size", 128))
            nl   = int(cfg.get("num_layers", 1))
            gate = 4
            total += gate * (cur * h + h * h + h)  # forward
            total += gate * (cur * h + h * h + h)  # backward
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
            total += cur * d + d  # proj
            total += nl * (4 * d * d + 4 * d + d * ff + ff + ff * d + d + 4 * d)
            cur = d
        elif lt == "residual":
            h = int(cfg.get("hidden_dim", 256))
            total += cur * h + h + h * cur + cur  # 2 linears
            if cfg.get("batchnorm"):
                total += 2 * h
            total += 2 * cur  # LayerNorm
            # cur unchanged
    # Output head
    total += cur * 1 + 1
    return total


# ── Architecture figure ───────────────────────────────────────────────────────

def _arch_figure(layer_configs: list, input_dim: int) -> plt.Figure:
    BG = "#F5F3EE"

    # Build display entries
    entries = [{"name": "Input", "dim": input_dim, "sub": f"concat(eA,eB)\n{input_dim}-dim"}]
    cur_dim = input_dim
    for i, cfg in enumerate(layer_configs):
        lt      = cfg["type"]
        out_dim = _compute_out_dim(lt, cur_dim, cfg)
        entries.append({
            "name": f"Layer {i + 1}",
            "dim":  out_dim,
            "sub":  lt.upper(),
        })
        cur_dim = out_dim
    entries.append({"name": "Output", "dim": 1, "sub": "sigmoid(logit)"})

    n      = len(entries)
    COLS   = ["#4A7BA5"] + ["#355E8E"] * (n - 2) + ["#2C4F73"]

    fig_w  = max(5.5, n * 1.65)
    fig, ax = plt.subplots(figsize=(fig_w, 3.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    max_dim  = max(e["dim"] for e in entries)
    MAX_H, MIN_H = 2.4, 0.45
    xs     = [(i + 0.5) * (fig_w / n) for i in range(n)]
    box_w  = (fig_w / n) * 0.52
    cy     = 1.9

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
                color="white", fontsize=9, fontweight="bold",
                fontfamily="monospace")

        ax.text(x, y0 + h + 0.1, entry["name"],
                ha="center", va="bottom",
                color="#1C1C1C", fontsize=8)

        ax.text(x, y0 - 0.1, entry["sub"],
                ha="center", va="top",
                color="#666666", fontsize=6.5, style="italic")

        if i < n - 1:
            x_next = xs[i + 1]
            ax.annotate(
                "", xy=(x_next - box_w / 2 - 0.06, cy),
                xytext=(x + box_w / 2 + 0.06, cy),
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.3),
            )

    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, 3.8)
    plt.tight_layout(pad=0.2)
    return fig


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(df, col_a, col_b, col_label):
    errors, warnings = [], []
    for col in [col_a, col_b, col_label]:
        n = df[col].isnull().sum()
        if n:
            errors.append(f"Column **{col}** has {n} missing value(s).")
    if errors:
        return errors, warnings, {}

    for col in [col_a, col_b]:
        n = (df[col].astype(str).str.strip() == "").sum()
        if n:
            errors.append(f"Column **{col}** has {n} empty sequence(s).")

    sample = df[[col_a, col_b]].head(200)
    for col in [col_a, col_b]:
        bad = sample[col].apply(
            lambda s: not bool(_VALID_AA.match(str(s).strip().upper()))
        )
        if bad.sum():
            first = sample[col][bad].iloc[0][:30]
            errors.append(
                f"Column **{col}** contains {int(bad.sum())} sequence(s) with invalid "
                f"characters (first: `{first}...`). Only amino acid letters accepted."
            )

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
                    f"(ratio {r:.1f}:1). Weighted loss will compensate, but consider "
                    f"adding more minority-class examples."
                )

    n_rows = len(df)
    if n_rows < 20:
        warnings.append(
            f"Only {n_rows} pairs. At least 100–200 pairs recommended for reliable training."
        )

    n_uniq = len(set(df[col_a].str.upper()) | set(df[col_b].str.upper()))
    stats  = {
        "rows":        n_rows,
        "unique_seqs": n_uniq,
        "n_pos":       int((df[col_label].astype(float).astype(int) == 1).sum()) if not errors else 0,
        "n_neg":       int((df[col_label].astype(float).astype(int) == 0).sum()) if not errors else 0,
    }
    return errors, warnings, stats


def _estimate_time(n_pairs: int, n_unique: int, mean_len: float,
                   epochs: int, batch_size: int) -> str:
    window  = max(1.0, mean_len / 512)
    embed_s = (n_unique / 8) * 0.25 * window
    train_s = epochs * math.ceil(n_pairs * 0.8 / batch_size) * 0.04
    total   = embed_s + train_s
    if total < 90:
        return f"~{int(total)} s"
    if total < 3600:
        return f"~{int(total / 60)} min"
    return f"~{total / 3600:.1f} h"


# =============================================================================
# Page
# =============================================================================

st.title("PPI Prediction")
st.caption("Build a sequence-based protein–protein interaction classifier.")
st.divider()

# ── 1. Training data ──────────────────────────────────────────────────────────
st.subheader("Training Data")

# Auto-load example data button
if st.button("Load Example Data", help="Loads 100 synthetic PPI pairs to test the workflow."):
    st.session_state["demo_loaded"] = True

col_dl, col_up = st.columns([1, 2])
with col_dl:
    st.download_button(
        "Download demo CSV", data=_DEMO_CSV,
        file_name="demo_ppi_train.csv", mime="text/csv",
        help="Download the synthetic demo CSV for offline inspection.",
    )
st.caption(
    "Upload a CSV with any column names. "
    "You will map them to **Protein A**, **Protein B**, and **Label (0/1)** below."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

# Determine data source
demo_loaded = st.session_state.get("demo_loaded", False)

if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        st.stop()
elif demo_loaded:
    st.info("Using example data (100 synthetic PPI pairs). Upload your own CSV to override.")
    raw_df = pd.read_csv(io.StringIO(_DEMO_CSV))
else:
    st.stop()

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

st.divider()

# ── 2. Column mapping ─────────────────────────────────────────────────────────
st.subheader("Column Mapping")
mc1, mc2, mc3 = st.columns(3)
with mc1:
    col_a     = st.selectbox("Protein A",   cols, index=0)
with mc2:
    col_b     = st.selectbox("Protein B",   cols, index=min(1, len(cols) - 1))
with mc3:
    col_label = st.selectbox("Label (0/1)", cols, index=min(2, len(cols) - 1))

if col_a == col_b:
    st.error("Protein A and Protein B cannot be the same column.")
    st.stop()

errors, warnings, stats = _validate(raw_df, col_a, col_b, col_label)

if errors:
    for e in errors:
        st.error(e)
    st.stop()

for w in warnings:
    st.warning(w)

seq_lens = pd.concat([
    raw_df[col_a].astype(str).str.len(),
    raw_df[col_b].astype(str).str.len(),
])
max_len  = int(seq_lens.max())
mean_len = float(seq_lens.mean())
n_long   = int((seq_lens > 1022).sum())

if max_len > 5000:
    st.warning(
        f"Very long sequences detected (max {max_len:,} residues). "
        f"Sliding-window embedding will be applied, significantly increasing GPU time."
    )
elif n_long > 0:
    st.info(f"{n_long:,} sequence(s) exceed 1022 residues — sliding-window embedding will be used.")

if stats["rows"] > 5000:
    st.warning(f"Large dataset ({stats['rows']:,} pairs). Consider enabling early stopping.")

st.success(
    f"Valid — {stats['rows']:,} pairs · {stats['unique_seqs']:,} unique sequences · "
    f"{stats['n_pos']:,} positive · {stats['n_neg']:,} negative · "
    f"avg length {int(mean_len):,} residues"
)

preview = raw_df[[col_a, col_b, col_label]].head(5).copy()
preview.columns = ["proteinA (preview)", "proteinB (preview)", "label"]
preview["proteinA (preview)"] = preview["proteinA (preview)"].astype(str).str[:50] + "…"
preview["proteinB (preview)"] = preview["proteinB (preview)"].astype(str).str[:50] + "…"
st.dataframe(preview, use_container_width=True, hide_index=True)

st.divider()

# ── 3. ESM2 model selection ───────────────────────────────────────────────────
st.subheader("Embedding Model")

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

st.caption(
    f"Model: `{esm_model_name}` · Embedding dimension: **{esm_dim}** · "
    f"Input to classifier: **{2 * esm_dim}** (concat of eA and eB)"
)

st.divider()

# ── 4. Model Builder ──────────────────────────────────────────────────────────
st.subheader("Model Builder")

# Initialise session state
st.session_state.setdefault("layers", [
    {"id": 0, "type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3, "batchnorm": False},
    {"id": 1, "type": "linear", "hidden_dim": 64,  "activation": "relu", "dropout": 0.2, "batchnorm": False},
])
st.session_state.setdefault("_lid", 2)

LAYER_TYPES     = ["linear", "cnn1d", "bilstm", "gru", "transformer", "residual"]
ACT_OPTIONS     = ["relu", "gelu", "tanh", "elu", "silu", "leaky_relu"]
KERNEL_OPTIONS  = [3, 5, 7, 9]
NHEAD_OPTIONS   = [2, 4, 8, 16]

layers: list = st.session_state["layers"]

# Render each layer
to_remove = None
to_move   = None  # (index, direction)  direction: "up" | "down"

for i, layer in enumerate(layers):
    lid = layer["id"]
    with st.container(border=True):
        hdr_cols = st.columns([5, 1, 1, 1])
        with hdr_cols[0]:
            st.markdown(f"**Layer {i + 1} — {layer['type'].upper()}**")
        with hdr_cols[1]:
            if i > 0:
                if st.button("↑", key=f"up_{lid}", help="Move up"):
                    to_move = (i, "up")
        with hdr_cols[2]:
            if i < len(layers) - 1:
                if st.button("↓", key=f"dn_{lid}", help="Move down"):
                    to_move = (i, "down")
        with hdr_cols[3]:
            if st.button("✕", key=f"rm_{lid}", help="Remove layer"):
                to_remove = i

        lt = layer["type"]

        if lt == "linear":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                layer["hidden_dim"] = st.number_input(
                    "hidden_dim", min_value=32, max_value=2048, step=32,
                    value=int(layer.get("hidden_dim", 256)), key=f"hd_{lid}",
                )
            with c2:
                layer["activation"] = st.selectbox(
                    "activation", ACT_OPTIONS,
                    index=ACT_OPTIONS.index(layer.get("activation", "relu")),
                    key=f"act_{lid}",
                )
            with c3:
                layer["dropout"] = st.slider(
                    "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                    key=f"drop_{lid}",
                )
            with c4:
                layer["batchnorm"] = st.checkbox(
                    "batchnorm", value=bool(layer.get("batchnorm", False)),
                    key=f"bn_{lid}",
                )

        elif lt == "cnn1d":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                layer["out_channels"] = st.number_input(
                    "out_channels", min_value=8, max_value=512, step=8,
                    value=int(layer.get("out_channels", 64)), key=f"och_{lid}",
                )
            with c2:
                ks_val = int(layer.get("kernel_size", 3))
                if ks_val not in KERNEL_OPTIONS:
                    ks_val = 3
                layer["kernel_size"] = st.selectbox(
                    "kernel_size", KERNEL_OPTIONS,
                    index=KERNEL_OPTIONS.index(ks_val), key=f"ks_{lid}",
                )
            with c3:
                layer["activation"] = st.selectbox(
                    "activation", ACT_OPTIONS,
                    index=ACT_OPTIONS.index(layer.get("activation", "relu")),
                    key=f"act_{lid}",
                )
            with c4:
                layer["dropout"] = st.slider(
                    "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                    key=f"drop_{lid}",
                )

        elif lt == "bilstm":
            c1, c2, c3 = st.columns(3)
            with c1:
                layer["hidden_size"] = st.number_input(
                    "hidden_size", min_value=32, max_value=512, step=32,
                    value=int(layer.get("hidden_size", 128)), key=f"hs_{lid}",
                )
            with c2:
                layer["num_layers"] = st.number_input(
                    "num_layers", min_value=1, max_value=3,
                    value=int(layer.get("num_layers", 1)), key=f"nl_{lid}",
                )
            with c3:
                layer["dropout"] = st.slider(
                    "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                    key=f"drop_{lid}",
                )

        elif lt == "gru":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                layer["hidden_size"] = st.number_input(
                    "hidden_size", min_value=32, max_value=512, step=32,
                    value=int(layer.get("hidden_size", 128)), key=f"hs_{lid}",
                )
            with c2:
                layer["num_layers"] = st.number_input(
                    "num_layers", min_value=1, max_value=3,
                    value=int(layer.get("num_layers", 1)), key=f"nl_{lid}",
                )
            with c3:
                layer["bidirectional"] = st.checkbox(
                    "bidirectional", value=bool(layer.get("bidirectional", True)),
                    key=f"bidir_{lid}",
                )
            with c4:
                layer["dropout"] = st.slider(
                    "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                    key=f"drop_{lid}",
                )

        elif lt == "transformer":
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                d_model_val = int(layer.get("d_model", 256))
                layer["d_model"] = st.number_input(
                    "d_model", min_value=64, max_value=1024, step=64,
                    value=d_model_val, key=f"dm_{lid}",
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
                    index=valid_nh.index(nhead_val), key=f"nh_{lid}",
                )
            with c3:
                layer["num_layers"] = st.number_input(
                    "num_layers", min_value=1, max_value=4,
                    value=int(layer.get("num_layers", 2)), key=f"nl_{lid}",
                )
            with c4:
                d_now = int(layer["d_model"])
                ff_opts = [d_now * 2, d_now * 4]
                ff_val  = int(layer.get("dim_feedforward", d_now * 2))
                if ff_val not in ff_opts:
                    ff_val = ff_opts[0]
                layer["dim_feedforward"] = st.selectbox(
                    "dim_feedforward", ff_opts,
                    index=ff_opts.index(ff_val), key=f"ff_{lid}",
                )
            with c5:
                layer["dropout"] = st.slider(
                    "dropout", 0.0, 0.5, float(layer.get("dropout", 0.1)), 0.05,
                    key=f"drop_{lid}",
                )

        elif lt == "residual":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                layer["hidden_dim"] = st.number_input(
                    "hidden_dim", min_value=32, max_value=2048, step=32,
                    value=int(layer.get("hidden_dim", 256)), key=f"hd_{lid}",
                )
            with c2:
                layer["activation"] = st.selectbox(
                    "activation", ACT_OPTIONS,
                    index=ACT_OPTIONS.index(layer.get("activation", "relu")),
                    key=f"act_{lid}",
                )
            with c3:
                layer["dropout"] = st.slider(
                    "dropout", 0.0, 0.7, float(layer.get("dropout", 0.3)), 0.05,
                    key=f"drop_{lid}",
                )
            with c4:
                layer["batchnorm"] = st.checkbox(
                    "batchnorm", value=bool(layer.get("batchnorm", False)),
                    key=f"bn_{lid}",
                )

# Apply remove / move mutations
if to_remove is not None:
    layers.pop(to_remove)
    st.session_state["layers"] = layers
    st.rerun()

if to_move is not None:
    idx, direction = to_move
    if direction == "up" and idx > 0:
        layers[idx], layers[idx - 1] = layers[idx - 1], layers[idx]
    elif direction == "down" and idx < len(layers) - 1:
        layers[idx], layers[idx + 1] = layers[idx + 1], layers[idx]
    st.session_state["layers"] = layers
    st.rerun()

# Add Layer section
st.markdown("---")
add_cols = st.columns([2, 1])
with add_cols[0]:
    new_type = st.selectbox("Layer type", LAYER_TYPES, key="new_layer_type", label_visibility="collapsed")
with add_cols[1]:
    if st.button("Add Layer", use_container_width=True):
        new_id = st.session_state["_lid"]
        st.session_state["_lid"] += 1
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
        st.session_state["layers"] = layers
        st.rerun()

st.divider()

# ── 5. Architecture visualization + param count ───────────────────────────────
input_dim = 2 * esm_dim
layer_configs_display = [
    {k: v for k, v in lyr.items() if k != "id"}
    for lyr in layers
]

col_viz, col_info = st.columns([3, 1])
with col_viz:
    st.caption("Architecture preview")
    if layers:
        fig_arch = _arch_figure(layer_configs_display, input_dim)
        st.pyplot(fig_arch, use_container_width=True)
        plt.close(fig_arch)
    else:
        st.info("Add at least one layer to preview the architecture.")

with col_info:
    if layers:
        n_param = _total_param_count(input_dim, layer_configs_display)
        st.metric("Approx. parameters", f"{n_param:,}")
    st.caption(f"Input dim: **{input_dim}**\n(2 × {esm_dim} from {esm_model_name})")

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

est = _estimate_time(stats["rows"], stats["unique_seqs"], mean_len, epochs, batch_size)
st.caption(
    f"**Estimated time:** {est} _(rough, GPU-dependent)_   |   "
    f"All jobs auto-stopped after **4 hours**."
)

st.divider()

# ── 7. Submit ─────────────────────────────────────────────────────────────────
if not layers:
    st.error("Add at least one layer before submitting.")
    st.stop()

if st.button("Submit Training Job", type="primary", use_container_width=True):
    hp = {
        "esm_model":           esm_model_name,
        "esm_dim":             esm_dim,
        "layer_configs":       layer_configs_display,
        "epochs":              epochs,
        "learning_rate":       lr,
        "batch_size":          batch_size,
        "early_stop_patience": early_stop,
    }
    send_df = raw_df[[col_a, col_b, col_label]].copy()
    send_df.columns = ["proteinA", "proteinB", "label"]
    send_df["label"] = send_df["label"].astype(float).astype(int)
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
            st.warning("**Save your cancel token — it will not be shown again.**")
            st.code(data["cancel_token"], language=None)
            st.info("Go to **Tools → Check Results** to monitor training progress.")
        except Exception as e:
            st.error(f"Submission failed: {e}")
