import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BACKEND = os.getenv("BACKEND_URL", "http://backend:8005")
is_dark = st.session_state.get("theme_mode", "Light") == "Dark"
plotly_template = st.session_state.get("plotly_template", "plotly_white")
card_bg = "#3a414b" if is_dark else "#f9f9f9"

MODEL_COLORS = ["#355E8E", "#E87040", "#2ecc71", "#9b59b6", "#f39c12"]
MAX_MODELS   = 5

TASK_LABELS = {
    "ppi": "PPI — Protein–Protein Interaction",
    "dtpi": "DTPI — Drug-Target Protein Interaction",
    "rpi": "RPI — RNA–Protein Interaction",
    "pdi": "PDI — Protein–DNA Interaction",
}

METRIC_ROWS = [
    ("val_acc",   "Val Accuracy"),
    ("auroc",     "AUROC"),
    ("ap",        "Avg Precision"),
    ("precision", "Precision"),
    ("recall",    "Recall"),
    ("f1",        "F1"),
    ("mcc",       "MCC"),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _approx_params(input_dim: int, layer_configs: list) -> int:
    total, cur = 0, input_dim
    for cfg in layer_configs:
        lt = cfg.get("type", "linear").lower()
        if lt == "linear":
            h = int(cfg.get("hidden_dim", 256))
            total += cur * h + h
            if cfg.get("batchnorm"):
                total += 2 * h
            cur = h
        elif lt == "cnn1d":
            out_ch = int(cfg.get("out_channels", 64))
            k      = int(cfg.get("kernel_size", 3))
            total += out_ch * k + out_ch
            cur = out_ch
        elif lt == "bilstm":
            h, gate = int(cfg.get("hidden_size", 128)), 4
            total += 2 * gate * (cur * h + h * h + h)
            cur = 2 * h
        elif lt == "gru":
            h     = int(cfg.get("hidden_size", 128))
            bidir = bool(cfg.get("bidirectional", True))
            dirs, gate = (2 if bidir else 1), 3
            total += dirs * gate * (cur * h + h * h + 2 * h)
            cur = dirs * h
        elif lt == "transformer":
            d  = int(cfg.get("d_model", 256))
            ff = int(cfg.get("dim_feedforward", d * 2))
            nl = int(cfg.get("num_layers", 2))
            total += cur * d + d + nl * (4 * d * d + 4 * d + d * ff + ff + ff * d + d + 4 * d)
            cur = d
        elif lt == "residual":
            h = int(cfg.get("hidden_dim", 256))
            total += cur * h + h + h * cur + cur
            if cfg.get("batchnorm"):
                total += 2 * h
            total += 2 * cur
    total += cur + 1
    return total


def _fetch_run(run_id: str) -> dict | None:
    """Fetch status + metrics for one run. Returns None on error."""
    try:
        sr = requests.get(f"{BACKEND}/check_status/{run_id}", timeout=5)
        if not sr.ok:
            return None
        status_data = sr.json()
        if "error" in status_data:
            return None
    except Exception:
        return None

    try:
        mr = requests.get(f"{BACKEND}/metrics/{run_id}", timeout=5)
        metrics_data = mr.json() if mr.ok else {}
    except Exception:
        metrics_data = {}

    return {"status": status_data, "metrics": metrics_data}


def _input_dim_from_hp(hp: dict, task_type: str) -> int:
    if task_type == "dtpi":
        return int(hp.get("chem_dim", 768)) + int(hp.get("esm_dim", 480))
    if task_type in ("rpi",):
        return int(hp.get("rna_dim", 640)) + int(hp.get("esm_dim", 480))
    if task_type in ("pdi",):
        return int(hp.get("dna_dim", 768)) + int(hp.get("esm_dim", 480))
    # ppi
    return int(hp.get("input_dim", int(hp.get("esm_dim", 480))))


def _arch_summary(hp: dict, task_type: str) -> str:
    layers = hp.get("layer_configs", [])
    if not layers:
        return "—"
    parts = []
    for cfg in layers:
        lt = cfg.get("type", "?").upper()
        if lt == "LINEAR":
            parts.append(f"Linear({cfg.get('hidden_dim','?')})")
        elif lt == "BILSTM":
            parts.append(f"BiLSTM({cfg.get('hidden_size','?')})")
        elif lt == "GRU":
            d = "Bi" if cfg.get("bidirectional", True) else ""
            parts.append(f"{d}GRU({cfg.get('hidden_size','?')})")
        elif lt == "CNN1D":
            parts.append(f"CNN1D({cfg.get('out_channels','?')}×{cfg.get('kernel_size','?')})")
        elif lt == "TRANSFORMER":
            parts.append(f"Transformer(d={cfg.get('d_model','?')},nl={cfg.get('num_layers','?')})")
        elif lt == "RESIDUAL":
            parts.append(f"Residual({cfg.get('hidden_dim','?')})")
        else:
            parts.append(lt)
    return " → ".join(parts) + " → Output(1)"


# ── page ──────────────────────────────────────────────────────────────────────

st.title("Model Comparison")
st.caption("Compare up to 5 trained models side by side.")
st.divider()

# ── Run ID management ─────────────────────────────────────────────────────────
if "cmp_run_ids" not in st.session_state:
    st.session_state["cmp_run_ids"] = []

with st.container():
    st.markdown("**Add run IDs to compare** (up to 5)")

    col_inp, col_add, col_clear = st.columns([3, 1, 1])
    with col_inp:
        new_id = st.text_input(
            "Run ID", label_visibility="collapsed",
            placeholder="Paste a run ID and click Add",
            key="cmp_new_id",
        )
    with col_add:
        add_disabled = len(st.session_state["cmp_run_ids"]) >= MAX_MODELS
        if st.button("Add", type="primary", disabled=add_disabled):
            rid_clean = (new_id or "").strip()
            if not rid_clean:
                st.warning("Enter a run ID first.")
            elif rid_clean in st.session_state["cmp_run_ids"]:
                st.warning("Run ID already in comparison.")
            else:
                st.session_state["cmp_run_ids"].append(rid_clean)
                st.rerun()
    with col_clear:
        if st.button("Clear All", type="secondary"):
            st.session_state["cmp_run_ids"] = []
            st.session_state.pop("cmp_data", None)
            st.rerun()

    # Display chips + remove buttons
    ids_now = st.session_state["cmp_run_ids"]
    if ids_now:
        chip_cols = st.columns(len(ids_now))
        remove_idx = None
        for i, rid in enumerate(ids_now):
            with chip_cols[i]:
                color = MODEL_COLORS[i % len(MODEL_COLORS)]
                st.markdown(
                    f'<span style="background:{color};color:white;padding:3px 10px;'
                    f'border-radius:12px;font-size:0.85em;font-weight:600">'
                    f'M{i+1}: {rid[:10]}…</span>',
                    unsafe_allow_html=True,
                )
                if st.button(f"Remove", key=f"cmp_rm_{i}"):
                    remove_idx = i
        if remove_idx is not None:
            st.session_state["cmp_run_ids"].pop(remove_idx)
            st.session_state.pop("cmp_data", None)
            st.rerun()

if not st.session_state["cmp_run_ids"]:
    st.info("Add at least two run IDs above to begin comparison.")
    st.stop()

# ── Fetch button ──────────────────────────────────────────────────────────────
st.divider()
if st.button("Load / Refresh Comparison", type="primary"):
    fetched = {}
    for rid in st.session_state["cmp_run_ids"]:
        with st.spinner(f"Fetching {rid} …"):
            data = _fetch_run(rid)
            if data is None:
                st.error(f"Could not load run `{rid}` — skipping.")
            else:
                fetched[rid] = data
    st.session_state["cmp_data"] = fetched

cmp_data: dict = st.session_state.get("cmp_data", {})
if not cmp_data:
    st.info("Click **Load / Refresh Comparison** to fetch data.")
    st.stop()

run_ids_loaded = [r for r in st.session_state["cmp_run_ids"] if r in cmp_data]
if len(run_ids_loaded) < 1:
    st.error("No valid runs loaded.")
    st.stop()

# ── Task type consistency check ───────────────────────────────────────────────
task_types = []
for rid in run_ids_loaded:
    hp = cmp_data[rid]["status"].get("hyperparams", {})
    tt = hp.get("task_type", cmp_data[rid]["status"].get("job_type", "ppi"))
    task_types.append(tt)

unique_tasks = list(dict.fromkeys(task_types))
if len(unique_tasks) > 1:
    st.warning(
        f"Mixed task types detected: {', '.join(unique_tasks).upper()}. "
        "Some charts may not be meaningful. Consider comparing only same-type models."
    )

dominant_task = unique_tasks[0]
st.markdown(f"**Task type:** `{TASK_LABELS.get(dominant_task, dominant_task.upper())}`")

# convenience aliases
def _color(i: int) -> str:
    return MODEL_COLORS[i % len(MODEL_COLORS)]

def _label(i: int, rid: str) -> str:
    return f"M{i+1}: {rid[:8]}"

# ── 1. Summary Metrics Table ──────────────────────────────────────────────────
st.divider()
st.subheader("Summary Metrics")

rows = []
for key, label in METRIC_ROWS:
    row = {"Metric": label}
    for i, rid in enumerate(run_ids_loaded):
        final = cmp_data[rid]["metrics"].get("final", {})
        v = final.get(key)
        row[_label(i, rid)] = f"{v:.4f}" if v is not None else "—"
    rows.append(row)

import pandas as pd
df_metrics = pd.DataFrame(rows).set_index("Metric")
st.dataframe(df_metrics, use_container_width=True)

# ── 2. Architecture Cards ─────────────────────────────────────────────────────
st.divider()
st.subheader("Architecture Overview")

arch_cols = st.columns(len(run_ids_loaded))
for i, rid in enumerate(run_ids_loaded):
    hp        = cmp_data[rid]["status"].get("hyperparams", {})
    tt        = task_types[run_ids_loaded.index(rid)]
    in_dim    = _input_dim_from_hp(hp, tt)
    layers    = hp.get("layer_configs", [])
    n_params  = _approx_params(in_dim, layers) if layers else None
    arch_str  = _arch_summary(hp, tt)
    color     = _color(i)

    with arch_cols[i]:
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:8px 12px;'
            f'background:{card_bg};border-radius:4px;margin-bottom:4px">'
            f'<b style="color:{color}">{_label(i, rid)}</b><br>'
            f'<small>Run: <code>{rid}</code></small></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Task: **{tt.upper()}**")
        st.caption(f"Input dim: **{in_dim:,}**")
        if n_params is not None:
            st.caption(f"~Params: **{n_params:,}**")

        # Embedding models
        if tt == "dtpi":
            chem = hp.get("chem_model", "ChemBERTa")
            esm  = hp.get("esm_model",  "ESM2")
            st.caption(f"Chem: `{chem.split('/')[-1]}`")
            st.caption(f"Prot: `{esm}`")
        elif tt == "rpi":
            rna = hp.get("rna_model", "RNA-FM")
            esm = hp.get("esm_model", "ESM2")
            st.caption(f"RNA:  `{rna.split('/')[-1] if '/' in rna else rna}`")
            st.caption(f"Prot: `{esm}`")
        elif tt == "pdi":
            dna = hp.get("dna_model", "DNABERT")
            esm = hp.get("esm_model", "ESM2")
            st.caption(f"DNA:  `{dna.split('/')[-1] if '/' in dna else dna}`")
            st.caption(f"Prot: `{esm}`")
        else:
            esm = hp.get("esm_model", "ESM2")
            pm  = hp.get("pair_mode", "all")
            st.caption(f"ESM2: `{esm}`")
            st.caption(f"Pair mode: `{pm}`")

        st.caption(f"Layers: {arch_str}")

# ── 3. Training Curves ────────────────────────────────────────────────────────
st.divider()
st.subheader("Training Curves")

has_history = any(
    cmp_data[rid]["metrics"].get("history", {}).get("epoch")
    for rid in run_ids_loaded
)

if has_history:
    fig_curves = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Validation Loss", "Validation Accuracy"],
    )
    for i, rid in enumerate(run_ids_loaded):
        hist = cmp_data[rid]["metrics"].get("history", {})
        epochs = hist.get("epoch", [])
        if not epochs:
            continue
        color = _color(i)
        name  = _label(i, rid)

        val_loss = hist.get("val_loss", [])
        val_acc  = hist.get("val_acc",  [])

        if val_loss:
            fig_curves.add_trace(
                go.Scatter(x=epochs, y=val_loss, mode="lines",
                           name=name, legendgroup=name,
                           line=dict(color=color, width=2)),
                row=1, col=1,
            )
        if val_acc:
            fig_curves.add_trace(
                go.Scatter(x=epochs, y=val_acc, mode="lines",
                           name=name, legendgroup=name,
                           showlegend=False,
                           line=dict(color=color, width=2)),
                row=1, col=2,
            )

    fig_curves.update_xaxes(title_text="Epoch")
    fig_curves.update_yaxes(title_text="Loss", row=1, col=1)
    fig_curves.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig_curves.update_layout(height=380, template=plotly_template,
                              legend=dict(title="Model"))
    st.plotly_chart(fig_curves, use_container_width=True)
else:
    st.caption("No training history available for any of the loaded runs.")

# ── 4. Overlaid ROC Curves ────────────────────────────────────────────────────
st.divider()
st.subheader("ROC & Precision-Recall Curves")

fig_roc_pr = make_subplots(
    rows=1, cols=2,
    subplot_titles=["ROC Curves", "Precision-Recall Curves"],
)

has_roc = False
has_pr  = False

# diagonal reference
fig_roc_pr.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
               line=dict(color="gray", dash="dash", width=1),
               showlegend=False, name="Random"),
    row=1, col=1,
)

for i, rid in enumerate(run_ids_loaded):
    mdata  = cmp_data[rid]["metrics"]
    final  = mdata.get("final", {})
    color  = _color(i)
    name   = _label(i, rid)
    auroc  = final.get("auroc")
    ap_val = final.get("ap")

    roc_data = mdata.get("roc_curve")
    if roc_data:
        fpr = [v for v in roc_data.get("fpr", []) if v is not None]
        tpr = [v for v in roc_data.get("tpr", []) if v is not None]
        if fpr and tpr:
            label_str = f"{name} (AUROC={auroc:.3f})" if auroc is not None else name
            fig_roc_pr.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines",
                           name=label_str, legendgroup=name,
                           line=dict(color=color, width=2)),
                row=1, col=1,
            )
            has_roc = True

    pr_data = mdata.get("pr_curve")
    if pr_data:
        prec = [v for v in pr_data.get("precision", []) if v is not None]
        rec  = [v for v in pr_data.get("recall",    []) if v is not None]
        if prec and rec:
            label_str = f"{name} (AP={ap_val:.3f})" if ap_val is not None else name
            fig_roc_pr.add_trace(
                go.Scatter(x=rec, y=prec, mode="lines",
                           name=label_str, legendgroup=name,
                           showlegend=not has_roc,
                           line=dict(color=color, width=2)),
                row=1, col=2,
            )
            has_pr = True

fig_roc_pr.update_xaxes(title_text="False Positive Rate", row=1, col=1, range=[0, 1])
fig_roc_pr.update_yaxes(title_text="True Positive Rate",  row=1, col=1, range=[0, 1.05])
fig_roc_pr.update_xaxes(title_text="Recall",    row=1, col=2, range=[0, 1])
fig_roc_pr.update_yaxes(title_text="Precision", row=1, col=2, range=[0, 1.05])
fig_roc_pr.update_layout(height=420, template=plotly_template,
                          legend=dict(title="Model"))

if has_roc or has_pr:
    st.plotly_chart(fig_roc_pr, use_container_width=True)
else:
    st.caption("ROC / PR curve data not available for any loaded run.")

# ── 5. Confusion Matrices ─────────────────────────────────────────────────────
cm_runs = [
    (i, rid) for i, rid in enumerate(run_ids_loaded)
    if cmp_data[rid]["metrics"].get("confusion_matrix") is not None
]

if cm_runs:
    st.divider()
    st.subheader("Confusion Matrices")
    n_cm = len(cm_runs)
    cm_cols = st.columns(max(n_cm, 3))[: n_cm]
    for col_idx, (i, rid) in enumerate(cm_runs):
        cm_data = cmp_data[rid]["metrics"]["confusion_matrix"]
        color   = _color(i)
        with cm_cols[col_idx]:
            try:
                cm_arr = np.array(cm_data, dtype=int)
                fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3.2))
                im = ax_cm.imshow(cm_arr, cmap="Blues", aspect="auto")
                plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
                ax_cm.set_xticks([0, 1])
                ax_cm.set_yticks([0, 1])
                ax_cm.set_xticklabels(["Pred 0", "Pred 1"])
                ax_cm.set_yticklabels(["True 0", "True 1"])
                ax_cm.set_title(_label(i, rid), color=color, fontweight="bold")
                for r in range(2):
                    for c in range(2):
                        v     = cm_arr[r, c]
                        tc    = "white" if v > cm_arr.max() / 2 else "black"
                        ax_cm.text(c, r, str(v), ha="center", va="center",
                                   color=tc, fontsize=13, fontweight="bold")
                plt.tight_layout(pad=0.3)
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)
            except Exception:
                st.caption(f"CM unavailable for {_label(i, rid)}")

# ── 6. Probability Distributions ─────────────────────────────────────────────
hist_runs = [
    (i, rid) for i, rid in enumerate(run_ids_loaded)
    if cmp_data[rid]["metrics"].get("prob_hist") is not None
]

if hist_runs:
    st.divider()
    st.subheader("Probability Distribution Overlay")

    fig_ph = go.Figure()
    for i, rid in hist_runs:
        hd     = cmp_data[rid]["metrics"]["prob_hist"]
        bins   = [v for v in hd.get("bins", []) if v is not None]
        counts = hd.get("counts", [])
        if bins and len(bins) == len(counts) + 1:
            centers = [(bins[j] + bins[j + 1]) / 2 for j in range(len(counts))]
            fig_ph.add_trace(go.Scatter(
                x=centers, y=counts, mode="lines+markers",
                name=_label(i, rid),
                line=dict(color=_color(i), width=2),
                marker=dict(size=4),
            ))
    fig_ph.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.6)
    fig_ph.update_layout(
        xaxis_title="Predicted Probability",
        yaxis_title="Count",
        height=340, template=plotly_template,
        legend=dict(title="Model"),
    )
    st.plotly_chart(fig_ph, use_container_width=True)
