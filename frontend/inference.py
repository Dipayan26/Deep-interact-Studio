"""
Inference page — /inference
Run a trained FlexiblePPIModel on new protein pairs, then visualise results
with an interactive dashboard: ROC/PR curves, KDE, scatter plot, confusion
matrix (when ground-truth labels are present), and a raw results table.
"""

import io
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

BACKEND = os.getenv("BACKEND_URL", "http://backend:8005")

# ── colour palette ────────────────────────────────────────────────────────────
C_POS   = "#185FA5"
C_NEG   = "#D85A30"
C_GREEN = "#1D9E75"
C_AMBER = "#BA7517"
BG      = "rgba(0,0,0,0)"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="sans-serif", size=12),
    margin=dict(l=50, r=20, t=36, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)

# =============================================================================
# Page header
# =============================================================================

st.title("Inference")
st.caption("Run a trained model on new protein pairs — then explore results interactively.")
st.divider()

# =============================================================================
# 1. Select trained model
# =============================================================================

st.subheader("Select Trained Model")

try:
    r    = requests.get(f"{BACKEND}/jobs", timeout=5)
    jobs = r.json()
    completed_training = [
        j for j in jobs
        if j.get("status") == "completed" and j.get("job_type", "train") == "train"
    ]
except Exception as e:
    st.error(f"Could not load jobs from backend: {e}")
    st.stop()

if not completed_training:
    st.info("No completed training jobs found. Train a model first on the Build page.")
    st.stop()

# ── task-type config ─────────────────────────────────────────────────────────
TASK_INPUT_CONFIGS = {
    "ppi": {
        "required_cols": ["proteinA", "proteinB"],
        "description":   (
            "Upload a CSV with columns **`proteinA`** and **`proteinB`** "
            "(amino acid sequences). "
            "Optionally include a **`label`** column (0/1) to unlock ROC/PR curves "
            "and a confusion matrix."
        ),
        "col_hint": "`proteinA`, `proteinB` · optional: `label`",
    },
    "dti": {
        "required_cols": ["protein", "drug"],
        "description":   (
            "Upload a CSV with columns **`protein`** (amino acid sequence) and "
            "**`drug`** (SMILES string). "
            "Optionally include a **`label`** column (0/1) for evaluation metrics."
        ),
        "col_hint": "`protein`, `drug` · optional: `label`",
    },
}
DEFAULT_TASK = "ppi"

job_map = {j["run_id"]: j for j in completed_training}

# ── Run ID input with validation ──────────────────────────────────────────────
last_used = st.session_state.get("last_run_id", "")
source_run_id = st.text_input(
    "Training Run ID",
    value=last_used,
    placeholder="e.g. abc123…",
    label_visibility="collapsed",
    help="Paste the Run ID of a completed training job.",
) or ""
source_run_id = source_run_id.strip()

if not source_run_id:
    st.info("Enter a training Run ID above to continue.")
    st.stop()

if source_run_id not in job_map:
    valid_ids = ", ".join(f"`{j['run_id']}`" for j in completed_training[-5:])
    st.error(
        f"Run ID `{source_run_id}` not found among completed training jobs. "
        f"Recent IDs: {valid_ids}"
    )
    st.stop()

selected_job = job_map[source_run_id]
task_type    = selected_job.get("task_type", DEFAULT_TASK)
task_cfg     = TASK_INPUT_CONFIGS.get(task_type, TASK_INPUT_CONFIGS[DEFAULT_TASK])

st.markdown(
    f"""
    <div style="
        display:inline-block; padding:6px 16px; border-radius:8px;
        background:#1a5fa520; border:1.5px solid #1a5fa560;
        font-size:0.92rem; font-weight:600; color:#1a5fa5; margin-bottom:4px;">
      {task_type.upper()}&nbsp;&nbsp;·&nbsp;&nbsp;
      acc&nbsp;=&nbsp;{selected_job.get('val_acc') or '—'}&nbsp;&nbsp;·&nbsp;&nbsp;
      auroc&nbsp;=&nbsp;{selected_job.get('auroc') or '—'}
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# =============================================================================
# 2. Input mode
# =============================================================================

st.session_state.setdefault("infer_input_key", 0)

hdr_c1, hdr_c2 = st.columns([5, 1])
with hdr_c1:
    st.subheader("Input Data")
with hdr_c2:
    if st.button("Reset", help="Clear all input fields.", use_container_width=True):
        st.session_state["infer_input_key"] += 1
        st.rerun()

input_mode = st.radio(
    "Mode",
    ["Single Pair", "Batch CSV"],
    horizontal=True,
    label_visibility="collapsed",
    key=f"input_mode_{st.session_state['infer_input_key']}",
)

st.divider()


def _parse_seq(raw: str) -> str:
    """Strip FASTA header(s) and return a clean uppercase sequence."""
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    seq_lines = [l for l in lines if not l.startswith(">")]
    return "".join(seq_lines).upper()


# =============================================================================
# 3a. Single-pair mode
# =============================================================================

if input_mode == "Single Pair":
    if task_type != "ppi":
        st.warning(f"Single-pair mode is currently only supported for PPI models.")
        st.stop()

    _ik = st.session_state["infer_input_key"]
    sp1, sp2 = st.columns(2)
    with sp1:
        raw_a = st.text_area(
            "Protein A",
            height=160,
            placeholder=">ProteinA (optional FASTA header)\nMKTAYIAKQ…",
            help="Paste a raw amino acid sequence or a FASTA block.",
            key=f"seq_a_{_ik}",
        )
    with sp2:
        raw_b = st.text_area(
            "Protein B",
            height=160,
            placeholder=">ProteinB (optional FASTA header)\nMSEQFLAG…",
            help="Paste a raw amino acid sequence or a FASTA block.",
            key=f"seq_b_{_ik}",
        )

    if st.button("Predict Interaction", type="primary", use_container_width=True):
        seq_a = _parse_seq(raw_a)
        seq_b = _parse_seq(raw_b)
        if not seq_a or not seq_b:
            st.error("Both sequences are required.")
        else:
            csv_bytes = f"proteinA,proteinB\n{seq_a},{seq_b}\n".encode()
            with st.spinner("Running inference…"):
                try:
                    r = requests.post(
                        f"{BACKEND}/run_inference/{source_run_id}",
                        files=[("files", ("pair.csv", csv_bytes, "text/csv"))],
                    )
                    r.raise_for_status()
                    data = r.json()
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        st.success(f"Job submitted — Run ID: `{data['run_id']}`")
                        st.session_state["infer_run_id"]     = data["run_id"]
                        st.session_state["infer_is_single"]  = True
                except Exception as e:
                    st.error(f"Submission failed: {e}")

# =============================================================================
# 3b. Batch CSV mode
# =============================================================================

else:
    st.markdown(task_cfg["description"])
    st.caption(f"Required columns: {task_cfg['col_hint']}")

    infer_file = st.file_uploader(
        "Select CSV file", type=["csv"], label_visibility="collapsed",
        key=f"infer_file_{st.session_state['infer_input_key']}",
    )

    if st.button("Run Inference", type="primary", use_container_width=True):
        if infer_file is None:
            st.error("No file selected.")
        else:
            try:
                infer_df = pd.read_csv(io.BytesIO(infer_file.getvalue()))
                missing  = [c for c in task_cfg["required_cols"] if c not in infer_df.columns]
                if missing:
                    st.error(
                        f"CSV is missing required column(s): "
                        f"{', '.join(f'`{c}`' for c in missing)}. "
                        f"Expected: {task_cfg['col_hint']}"
                    )
                    st.stop()
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
                st.stop()

            with st.spinner("Submitting inference job…"):
                try:
                    files = [("files", (infer_file.name, infer_file.getvalue(), "text/csv"))]
                    r = requests.post(f"{BACKEND}/run_inference/{source_run_id}", files=files)
                    r.raise_for_status()
                    data = r.json()
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        st.success(f"Inference job submitted — Run ID: `{data['run_id']}`")
                        st.session_state["infer_run_id"]    = data["run_id"]
                        st.session_state["infer_is_single"] = False
                except Exception as e:
                    st.error(f"Submission failed: {e}")

# =============================================================================
# 4. Poll status
# =============================================================================

infer_run_id = st.session_state.get("infer_run_id", "")
if not infer_run_id:
    st.stop()

st.divider()
st.subheader("Results")

rid_input = st.text_input("Inference Run ID", value=infer_run_id or "")
rid       = (rid_input or "").strip()

col_check, col_auto = st.columns([1, 3])
with col_check:
    check_btn = st.button("Check Status", key="infer_check")
with col_auto:
    auto = st.checkbox("Auto-refresh", value=False, key="infer_auto")

if not (check_btn or auto):
    st.stop()

try:
    sr = requests.get(f"{BACKEND}/check_status/{rid}", timeout=5)
    sd = sr.json()
except Exception as e:
    st.error(f"Could not reach backend: {e}")
    st.stop()

if "error" in sd:
    st.error("Run ID not found.")
    st.stop()

status = sd["status"]
colour = {"completed": "green", "running": "blue",
          "queued": "orange", "failed": "red"}.get(status, "gray")
st.markdown(f"**Status:** :{colour}[{status.capitalize()}]")

if status == "failed":
    st.error(f"Inference failed: {sd.get('result', 'check backend logs')}")
    st.stop()

if status in ("running", "queued"):
    st.info("Inference in progress...")
    if auto:
        time.sleep(4)
        st.rerun()
    st.stop()

# =============================================================================
# 5. Load results
# =============================================================================

resp_csv = requests.get(f"{BACKEND}/download_results/{rid}", stream=True)
if resp_csv.status_code != 200:
    st.warning("Results file not available yet.")
    st.stop()

results_df = pd.read_csv(io.BytesIO(resp_csv.content))

# Rich metrics from /inference_metrics endpoint
try:
    mr          = requests.get(f"{BACKEND}/inference_metrics/{rid}", timeout=5)
    inf_metrics = mr.json() if mr.ok else {}
except Exception:
    inf_metrics = {}

has_labels = inf_metrics.get("has_labels", False)
probs      = np.array(inf_metrics.get("probabilities",
             results_df["probability"].tolist() if "probability" in results_df else []))
labels     = np.array(inf_metrics.get("labels", [])) if has_labels else None

# =============================================================================
# 6. Summary + dashboard
# =============================================================================

is_single = st.session_state.get("infer_is_single", False)

n_pairs    = len(results_df)
n_pos_pred = int((results_df.get("prediction", pd.Series(dtype=int)) == 1).sum())
n_neg_pred = n_pairs - n_pos_pred
mean_prob  = float(probs.mean()) if len(probs) else 0.0

# ── Single-pair result card ───────────────────────────────────────────────────
if is_single and n_pairs == 1:
    prob_val = float(probs[0]) if len(probs) else mean_prob
    pred_val = int(results_df["prediction"].iloc[0]) if "prediction" in results_df.columns else int(prob_val >= 0.5)
    label    = "Interacting" if pred_val == 1 else "Not Interacting"
    colour   = C_POS if pred_val == 1 else C_NEG

    st.markdown(
        f"""
        <div style="
            border-radius:12px; padding:28px 36px; margin:12px 0;
            background:linear-gradient(135deg,{colour}18,{colour}08);
            border:2px solid {colour}55; text-align:center;">
          <div style="font-size:2rem; font-weight:700; color:{colour};">{label}</div>
          <div style="font-size:1.1rem; color:#555; margin-top:6px;">
            Interaction probability: <strong>{prob_val:.4f}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download result (.csv)",
        data=resp_csv.content,
        file_name=f"ppi_result_{rid}.csv",
        mime="text/csv",
    )

# ── Batch result cards + full dashboard ──────────────────────────────────────
else:
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Pairs scored",    f"{n_pairs:,}")
    mc2.metric("Predicted +",     f"{n_pos_pred:,}")
    mc3.metric("Predicted −",     f"{n_neg_pred:,}")
    mc4.metric("Mean probability", f"{mean_prob:.3f}")

    if has_labels and inf_metrics.get("auroc") is not None:
        st.divider()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AUROC",    f"{inf_metrics['auroc']:.4f}")
        m2.metric("AUPRC",    f"{inf_metrics.get('auprc', 0):.4f}")
        m3.metric("F1",       f"{inf_metrics.get('f1', 0):.4f}")
        m4.metric("Accuracy", f"{inf_metrics.get('accuracy', 0):.4f}")
        m5.metric("MCC",      f"{inf_metrics.get('mcc', 0):.4f}")

    st.divider()

    st.download_button(
        "Download results (.csv)",
        data=resp_csv.content,
        file_name=f"ppi_results_{rid}.csv",
        mime="text/csv",
    )

    st.divider()

    # ── Dashboard tabs ────────────────────────────────────────────────────────
    tabs_list = ["Probability Distribution", "Score Scatter", "Raw Results"]
    if has_labels:
        tabs_list = ["ROC & PR Curves", "Confusion Matrix"] + tabs_list

    tabs = st.tabs(tabs_list)
    tab_offset = 2 if has_labels else 0

    # ── helper: KDE ───────────────────────────────────────────────────────────
    def _kde(data: np.ndarray, bw: float = 0.04, n: int = 200):
        xs = np.linspace(0, 1, n)
        ys = np.array([
            np.mean(np.exp(-0.5 * ((x - data) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)))
            for x in xs
        ])
        return xs, ys

    # ROC & PR (label-gated) ──────────────────────────────────────────────────
    if has_labels and labels is not None and len(labels):
        from sklearn.metrics import (
            roc_curve, auc, precision_recall_curve,
            confusion_matrix, matthews_corrcoef,
        )
        fpr, tpr, _      = roc_curve(labels, probs)
        roc_auc          = auc(fpr, tpr)
        prec_a, rec_a, _ = precision_recall_curve(labels, probs)
        pr_auc           = auc(rec_a, prec_a)

        with tabs[0]:
            st.markdown("##### ROC & Precision–Recall curves")
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("ROC curve", "Precision–Recall curve"))
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                name=f"AUROC = {roc_auc:.3f}", line=dict(color=C_POS, width=2),
                fill="tozeroy", fillcolor="rgba(24,95,165,0.09)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                line=dict(color=C_NEG, dash="dash", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=rec_a, y=prec_a, mode="lines",
                name=f"AUPRC = {pr_auc:.3f}", line=dict(color=C_GREEN, width=2),
                fill="tozeroy", fillcolor="rgba(29,158,117,0.09)"), row=1, col=2)
            fig.update_xaxes(title_text="False positive rate", range=[0,1], row=1, col=1)
            fig.update_yaxes(title_text="True positive rate",  range=[0,1], row=1, col=1)
            fig.update_xaxes(title_text="Recall",    range=[0,1], row=1, col=2)
            fig.update_yaxes(title_text="Precision", range=[0,1], row=1, col=2)
            fig.update_layout(**PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            st.markdown("##### Confusion matrix & per-class metrics")
            thr_cm = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01, key="cm_thr")
            preds  = (probs >= thr_cm).astype(int)
            cm     = confusion_matrix(labels, preds)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            cm_fig = go.Figure(go.Heatmap(
                z=[[tp, fp], [fn, tn]],
                x=["Pred Positive", "Pred Negative"],
                y=["True Positive", "True Negative"],
                text=[[str(tp), str(fp)], [str(fn), str(tn)]],
                texttemplate="%{text}", textfont=dict(size=22),
                colorscale=[[0, "#f5f5f3"], [1, C_POS]], showscale=False,
            ))
            cm_fig.update_layout(**PLOTLY_LAYOUT, height=300,
                                 xaxis=dict(side="top"), yaxis=dict(autorange="reversed"))
            st.plotly_chart(cm_fig, use_container_width=True)
            prec_v = tp/(tp+fp) if (tp+fp) else 0
            rec_v  = tp/(tp+fn) if (tp+fn) else 0
            spec_v = tn/(tn+fp) if (tn+fp) else 0
            acc_v  = (tp+tn)/len(labels) if len(labels) else 0
            f1_v   = 2*prec_v*rec_v/(prec_v+rec_v) if (prec_v+rec_v) else 0
            mcc_v  = matthews_corrcoef(labels, preds)
            st.dataframe(pd.DataFrame({
                "Metric": ["Accuracy","Precision","Recall / Sensitivity",
                           "Specificity","F1","MCC","AUROC","AUPRC"],
                "Value":  [f"{v:.4f}" for v in
                           [acc_v,prec_v,rec_v,spec_v,f1_v,mcc_v,roc_auc,pr_auc]],
            }), use_container_width=True, hide_index=True)

    # Probability distribution ────────────────────────────────────────────────
    with tabs[tab_offset]:
        st.markdown("##### Predicted probability distribution")
        thr_kde = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01, key="kde_thr")
        fig_kde = go.Figure()
        if has_labels and labels is not None and len(labels):
            neg_p = probs[labels == 0]
            pos_p = probs[labels == 1]
            if len(neg_p):
                xs, ys = _kde(neg_p)
                fig_kde.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                    name="Negative (label=0)", line=dict(color=C_NEG, width=2),
                    fill="tozeroy", fillcolor="rgba(216,90,48,0.12)"))
            if len(pos_p):
                xs, ys = _kde(pos_p)
                fig_kde.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                    name="Positive (label=1)", line=dict(color=C_POS, width=2),
                    fill="tozeroy", fillcolor="rgba(24,95,165,0.12)"))
        else:
            xs, ys = _kde(probs)
            fig_kde.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="All pairs",
                line=dict(color=C_POS, width=2),
                fill="tozeroy", fillcolor="rgba(24,95,165,0.12)"))
        fig_kde.add_vline(x=thr_kde, line_dash="dash", line_color=C_AMBER, line_width=2,
                          annotation_text=f"thr={thr_kde:.2f}",
                          annotation_position="top right",
                          annotation_font_color=C_AMBER)
        fig_kde.update_xaxes(title_text="Predicted probability", range=[0,1])
        fig_kde.update_yaxes(title_text="Density")
        fig_kde.update_layout(**PLOTLY_LAYOUT, height=340)
        st.plotly_chart(fig_kde, use_container_width=True)

    # Score scatter ───────────────────────────────────────────────────────────
    with tabs[tab_offset + 1]:
        st.markdown("##### Probability scatter — pair index vs. score")
        st.caption("Each point is one pair. Colour = predicted probability. "
                   "When labels are present, filled circle = positive, open = negative.")
        scatter_df = results_df.copy()
        scatter_df["idx"] = range(len(scatter_df))
        if "probability" not in scatter_df.columns:
            scatter_df["probability"] = probs
        short_a = scatter_df["proteinA"].astype(str).str[:20] + "…"
        short_b = scatter_df["proteinB"].astype(str).str[:20] + "…"
        scatter_df["hover"] = short_a + " × " + short_b
        if has_labels and labels is not None and len(labels) == len(scatter_df):
            scatter_df["true_label"] = labels.astype(int).astype(str)
            fig_sc = px.scatter(scatter_df, x="idx", y="probability",
                color="probability", color_continuous_scale=[C_NEG, C_POS],
                symbol="true_label", symbol_map={"0": "circle-open", "1": "circle"},
                hover_name="hover",
                hover_data={"idx": False, "probability": ":.3f", "true_label": True},
                labels={"idx": "Pair index", "probability": "Probability",
                        "true_label": "True label"})
        else:
            fig_sc = px.scatter(scatter_df, x="idx", y="probability",
                color="probability", color_continuous_scale=[C_NEG, C_POS],
                hover_name="hover",
                hover_data={"idx": False, "probability": ":.3f"},
                labels={"idx": "Pair index", "probability": "Probability"})
        thr_sc = st.slider("Highlight cut-off", 0.0, 1.0, 0.5, 0.01, key="sc_thr")
        fig_sc.add_hline(y=thr_sc, line_dash="dash", line_color=C_AMBER, line_width=1.5)
        fig_sc.update_traces(marker=dict(size=7, opacity=0.75))
        fig_sc.update_layout(**PLOTLY_LAYOUT, height=360,
                             coloraxis_colorbar=dict(title="P(interact)"))
        st.plotly_chart(fig_sc, use_container_width=True)
        n_above = int((scatter_df["probability"] >= thr_sc).sum())
        st.caption(f"{n_above} / {n_pairs} pairs ≥ {thr_sc:.2f}")

    # Raw results table ───────────────────────────────────────────────────────
    with tabs[tab_offset + 2]:
        st.markdown("##### All scored pairs")
        search  = st.text_input("Filter by sequence substring", placeholder="e.g. MKTAY…")
        show_df = results_df.copy()
        if search.strip():
            mask = (
                show_df["proteinA"].astype(str).str.contains(search, case=False, na=False) |
                show_df["proteinB"].astype(str).str.contains(search, case=False, na=False)
            )
            show_df = show_df[mask]
        for col in ("proteinA", "proteinB"):
            if col in show_df.columns:
                show_df[col] = show_df[col].astype(str).str[:40] + "…"
        st.dataframe(show_df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(show_df):,} of {n_pairs:,} rows")

        st.divider()
        st.markdown("##### Threshold sensitivity")
        thr_rows = []
        for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            n = int((results_df.get("probability", pd.Series(probs)) >= t).sum())
            thr_rows.append({"Threshold": t, "Predicted positive": n,
                             "Predicted negative": n_pairs - n})
        st.dataframe(pd.DataFrame(thr_rows), use_container_width=True, hide_index=True)
