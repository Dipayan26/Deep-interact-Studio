import io
import json
import math
import os
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

_DEMO_CSV = """\
proteinA,proteinB,label
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL,MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSTNEHGSEHTVDGVDGENFEDVSQDIFQTYFLTLNQTSDGDVDAIWGIALGGLQHMDKDAGAAKWESGYDEFGGALFRGDVFSNMHTYFNKIYSEREENLEKYGDVAQVSDENKNWYKAACVGAMMQMLKSHMTNAVLKVMREAGKKDAVDKMLQQLNQLEKERMAAQMKDLVSAVAQHTSSMVGKGSFEDSLVDYGIRTALMESLGGTPNKSTKEELDKYFKNHTSPDVDGDLGDHLSDYFGKLCVLYGDGIVLGTGSRQNPTQAELRDVSRAFAATLVSGLRTLACFAEHRGDTDVLKELLEKVHERDHGDENLKIIVQDTVSDIITQGKIPVEQFISAFKIVNDGVIKVLREHNAQNTPKEQLIEMFKRMHNSFAAVPSHSSTGGKFNYSGVR,1
MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL,MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL,0
MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSTNEHGSEHTVDGVDGENFEDVSQDIFQTYFLTLNQTSDGDVDAIWGIALGGLQHMDKDAGAAKWESGYDEFGGALFRGDVFSNMHTYFNKIYSEREENLEKYGDVAQVSDENKNWYKAACVGAMMQMLKSHMTNAVLKVMREAGKKDAVDKMLQQLNQLEKERMAAQMKDLVSAVAQHTSSMVGKGSFEDSLVDYGIRTALMESLGGTPNKSTKEELDKYFKNHTSPDVDGDLGDHLSDYFGKLCVLYGDGIVLGTGSRQNPTQAELRDVSRAFAATLVSGLRTLACFAEHRGDTDVLKELLEKVHERDHGDENLKIIVQDTVSDIITQGKIPVEQFISAFKIVNDGVIKVLREHNAQNTPKEQLIEMFKRMHNSFAAVPSHSSTGGKFNYSGVR,MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL,1
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL,MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL,0
"""

ESM2_DIM = 480

# ── helpers ───────────────────────────────────────────────────────────────────

def _input_dim(pair_method: str) -> int:
    return {"concat": ESM2_DIM * 2, "product": ESM2_DIM,
            "diff": ESM2_DIM, "all": ESM2_DIM * 4}[pair_method]


def _param_count(input_dim: int, hidden_dim: int, num_layers: int) -> int:
    total = input_dim * hidden_dim + hidden_dim
    for _ in range(num_layers - 1):
        total += hidden_dim * hidden_dim + hidden_dim
    total += hidden_dim * 2 + 2
    return total


def _estimate_time(n_pairs: int, n_unique: int, mean_len: float,
                   epochs: int, batch_size: int) -> str:
    window  = max(1.0, mean_len / 512)
    embed_s = (n_unique / 8) * 0.25 * window          # ~0.25 s/batch on GPU
    train_s = epochs * math.ceil(n_pairs * 0.8 / batch_size) * 0.04
    total   = embed_s + train_s
    if total < 90:
        return f"~{int(total)} s"
    if total < 3600:
        return f"~{int(total / 60)} min"
    return f"~{total / 3600:.1f} h"


def _model_figure(input_dim: int, hidden_dim: int, num_layers: int,
                  pair_method: str) -> plt.Figure:
    BG   = "#F5F3EE"
    COLS = ["#4A7BA5"] + ["#355E8E"] * num_layers + ["#2C4F73"]
    PAIR_LABELS = {
        "concat":  "eA ⊕ eB",
        "product": "eA ⊙ eB",
        "diff":    "|eA − eB|",
        "all":     "eA ⊕ eB ⊕ eA⊙eB ⊕ |eA−eB|",
    }

    layers = (
        [{"name": "Input", "dim": input_dim,  "sub": PAIR_LABELS[pair_method]}]
        + [{"name": f"Hidden {i+1}", "dim": hidden_dim, "sub": "ReLU · Dropout(0.3)"}
           for i in range(num_layers)]
        + [{"name": "Output", "dim": 2, "sub": "Softmax"}]
    )

    n      = len(layers)
    fig_w  = max(5.5, n * 1.55)
    fig, ax = plt.subplots(figsize=(fig_w, 3.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    max_dim    = max(l["dim"] for l in layers)
    MAX_H, MIN_H = 2.4, 0.45
    xs         = [(i + 0.5) * (fig_w / n) for i in range(n)]
    box_w      = (fig_w / n) * 0.52
    cy         = 1.9                          # vertical center

    for i, (layer, x, col) in enumerate(zip(layers, xs, COLS)):
        h = MIN_H + (MAX_H - MIN_H) * (
            math.log2(layer["dim"] + 1) / math.log2(max_dim + 1)
        )
        y0 = cy - h / 2

        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y0), box_w, h,
            boxstyle="round,pad=0.05",
            facecolor=col, edgecolor="white", linewidth=1.2,
        )
        ax.add_patch(rect)

        # dimension number (bold, white, inside box)
        ax.text(x, cy, f"{layer['dim']:,}",
                ha="center", va="center",
                color="white", fontsize=10, fontweight="bold",
                fontfamily="monospace")

        # layer name above box
        ax.text(x, y0 + h + 0.1, layer["name"],
                ha="center", va="bottom",
                color="#1C1C1C", fontsize=8)

        # sublabel below box
        ax.text(x, y0 - 0.1, layer["sub"],
                ha="center", va="top",
                color="#666666", fontsize=6.5, style="italic")

        # arrow →
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
        "rows": n_rows, "unique_seqs": n_uniq,
        "n_pos": int((df[col_label].astype(float).astype(int) == 1).sum()) if not errors else 0,
        "n_neg": int((df[col_label].astype(float).astype(int) == 0).sum()) if not errors else 0,
    }
    return errors, warnings, stats


# =============================================================================
st.title("PPI Prediction")
st.caption("Build a sequence-based protein–protein interaction classifier.")
st.divider()

# ── 1. Training data ──────────────────────────────────────────────────────────
st.subheader("Training Data")

st.download_button(
    "Download demo CSV", data=_DEMO_CSV,
    file_name="demo_ppi_train.csv", mime="text/csv",
    help="Sample file to test a run before uploading your own data.",
)
st.caption(
    "Upload a CSV with any column names. "
    "You will map them to **Protein A**, **Protein B**, and **Label (0/1)** below."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

if uploaded is None:
    st.stop()

try:
    raw_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not parse CSV: {e}")
    st.stop()

if raw_df.empty:
    st.error("The uploaded file is empty.")
    st.stop()

cols = raw_df.columns.tolist()

# raw preview
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
    col_b     = st.selectbox("Protein B",   cols, index=min(1, len(cols)-1))
with mc3:
    col_label = st.selectbox("Label (0/1)", cols, index=min(2, len(cols)-1))

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

# sequence length info
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

# ── 3. Model configuration ────────────────────────────────────────────────────
st.subheader("Model Configuration")

col_hp, col_viz = st.columns([1, 1], gap="large")

with col_hp:
    pair_method = st.selectbox(
        "Pair representation",
        ["all", "concat", "product", "diff"],
        help="How the two protein embeddings are combined into a single vector.",
    )
    hidden_dim  = st.selectbox("Hidden layer size", [128, 256, 512], index=1)
    num_layers  = st.selectbox("MLP depth (layers)", [2, 3], index=0)
    batch_size  = st.selectbox("Batch size", [32, 64, 128], index=1)
    epochs      = st.slider("Epochs", min_value=5, max_value=100, value=30, step=5)
    lr          = st.selectbox(
        "Learning rate", [0.001, 0.0005, 0.0001],
        format_func=lambda x: f"{x:.4f}",
    )
    early_stop  = st.selectbox(
        "Early stopping patience",
        [0, 5, 10, 15],
        format_func=lambda x: "Disabled" if x == 0 else f"{x} epochs without improvement",
        index=1,
    )

with col_viz:
    in_dim  = _input_dim(pair_method)
    n_param = _param_count(in_dim, hidden_dim, num_layers)
    est     = _estimate_time(stats["rows"], stats["unique_seqs"],
                              mean_len, epochs, batch_size)

    st.caption("Model architecture")
    fig = _model_figure(in_dim, hidden_dim, num_layers, pair_method)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        f"**Trainable parameters:** {n_param:,}    "
        f"**Estimated time:** {est} _(rough, GPU-dependent)_"
    )
    st.caption(f"**Input dimension:** {in_dim} ({pair_method} pair representation of {ESM2_DIM}-dim ESM2 embeddings)")

st.caption("All jobs are automatically stopped after **4 hours**.")
st.divider()

# ── 4. Submit ─────────────────────────────────────────────────────────────────
if st.button("Submit Training Job", type="primary", use_container_width=True):
    hp = {
        "pair_method":         pair_method,
        "hidden_dim":          hidden_dim,
        "num_layers":          num_layers,
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

            st.session_state["last_run_id"]       = data["run_id"]
            st.session_state["last_cancel_token"]  = data["cancel_token"]

            st.success(f"Job submitted — Run ID: `{data['run_id']}`")
            st.warning(
                "**Save your cancel token — it will not be shown again.**"
            )
            st.code(data["cancel_token"], language=None)
            st.info(
                "Go to **Tools → Check Results** to monitor training progress."
            )
        except Exception as e:
            st.error(f"Submission failed: {e}")
