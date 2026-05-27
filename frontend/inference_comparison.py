"""
Inference Comparison — compare up to 5 completed inference runs side by side.

Requirements:
  • All selected runs must be the same task type (PPI / DTPI / RPI / PDI).
  • All selected runs should have been run on the same input data for
    results to be meaningful — the page warns when source training runs differ.
  • Runs are selected either from Job Status (checkbox → "Compare Inferences")
    or by pasting run IDs directly here.

Data source: GET /inference_metrics/{run_id}
  Returns: {has_labels, probabilities, labels?, auroc?, auprc?, f1?,
            accuracy?, mcc?, roc_curve?, pr_curve?, confusion_matrix?,
            prob_hist?}
"""

import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from model_details import render_layer_difference_table, render_model_details

BACKEND        = os.getenv("BACKEND_URL", "http://backend:8005")
is_dark        = st.session_state.get("theme_mode", "Light") == "Dark"
plotly_template = st.session_state.get("plotly_template", "plotly_white")
card_bg        = "#3a414b" if is_dark else "#f9f9f9"

MODEL_COLORS = ["#355E8E", "#E87040", "#2ecc71", "#9b59b6", "#f39c12"]
MAX_RUNS     = 5

TASK_LABELS = {
    "ppi":  "PPI — Protein–Protein Interaction",
    "dtpi": "DTPI — Drug-Target Protein Interaction",
    "rpi":  "RPI — RNA–Protein Interaction",
    "pdi":  "PDI — Protein–DNA Interaction",
}

METRIC_ROWS = [
    ("accuracy", "Accuracy"),
    ("auroc",    "AUROC"),
    ("auprc",    "Avg Precision (AUPRC)"),
    ("f1",       "F1"),
    ("mcc",      "MCC"),
]

# =============================================================================
# Helpers
# =============================================================================

def _color(i: int) -> str:
    return MODEL_COLORS[i % len(MODEL_COLORS)]


def _label(i: int, rid: str) -> str:
    return f"Run {i+1}: {rid[:10]}"


def _fetch_inference(run_id: str) -> dict | None:
    """Fetch check_status + job detail + inference_metrics for one inference run."""
    try:
        sr = requests.get(f"{BACKEND}/check_status/{run_id}", timeout=5)
        if not sr.ok:
            return None
        sd = sr.json()
        if "error" in sd:
            return None
        if sd.get("job_type", sd.get("status", "")) not in (
            "inference", "completed", "running", "queued", "failed"
        ):
            pass  # job_type may be absent in older records
    except Exception:
        return None

    try:
        jd = requests.get(f"{BACKEND}/job_detail/{run_id}", timeout=5).json()
        if jd.get("source_run_id") and not sd.get("source_run_id"):
            sd["source_run_id"] = jd["source_run_id"]
        if jd.get("hyperparams") and not sd.get("hyperparams"):
            sd["hyperparams"] = jd["hyperparams"]
    except Exception:
        jd = {}

    try:
        mr       = requests.get(f"{BACKEND}/inference_metrics/{run_id}", timeout=5)
        inf_data = mr.json() if mr.ok else {}
    except Exception:
        inf_data = {}

    return {"status": sd, "detail": jd, "metrics": inf_data}


def _fetch_training_detail(run_id: str) -> dict:
    if not run_id:
        return {}
    try:
        jr = requests.get(f"{BACKEND}/job_detail/{run_id}", timeout=5)
        detail = jr.json() if jr.ok else {}
    except Exception:
        detail = {}
    try:
        mr = requests.get(f"{BACKEND}/metrics/{run_id}", timeout=5)
        detail["metrics"] = mr.json() if mr.ok else {}
    except Exception:
        detail["metrics"] = {}
    return detail


def _kde(data: np.ndarray, bw: float = 0.04, n: int = 200) -> tuple:
    xs = np.linspace(0, 1, n)
    ys = np.array([
        np.mean(np.exp(-0.5 * ((x - data) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)))
        for x in xs
    ])
    return xs, ys


# =============================================================================
# Page header
# =============================================================================

st.title("Inference Comparison")
st.markdown(
    "**Compare up to 5 inference runs across predictions, metrics, score distributions, and threshold behavior. "
    "Use Job Status to select compatible completed inference runs or paste run IDs below; best results come from the same task and input dataset.**"
)
st.divider()

# =============================================================================
# Run ID management
# =============================================================================

if "icmp_run_ids" not in st.session_state:
    # pre-populate from job_status selection if available
    js_sel = st.session_state.get("js_icmp_selected", [])
    st.session_state["icmp_run_ids"] = js_sel[:MAX_RUNS] if js_sel else []

st.markdown("**Add inference run IDs** (up to 5)")

col_inp, col_add, col_clear = st.columns([3, 1, 1])
with col_inp:
    new_id = st.text_input(
        "Run ID", label_visibility="collapsed",
        placeholder="Paste an inference run ID and click Add",
        key="icmp_new_id",
    )
with col_add:
    add_disabled = len(st.session_state["icmp_run_ids"]) >= MAX_RUNS
    if st.button("Add", type="primary", disabled=add_disabled):
        rid_clean = (new_id or "").strip()
        if not rid_clean:
            st.warning("Enter a run ID first.")
        elif rid_clean in st.session_state["icmp_run_ids"]:
            st.warning("Run ID already in comparison.")
        elif len(st.session_state["icmp_run_ids"]) >= MAX_RUNS:
            st.warning(f"Maximum {MAX_RUNS} runs allowed.")
        else:
            st.session_state["icmp_run_ids"].append(rid_clean)
            st.session_state.pop("icmp_data", None)
            st.rerun()
with col_clear:
    if st.button("Clear All", type="secondary"):
        st.session_state["icmp_run_ids"] = []
        st.session_state.pop("icmp_data", None)
        st.session_state["js_icmp_selected"] = []
        st.rerun()

# Chips
ids_now = st.session_state["icmp_run_ids"]
if ids_now:
    chip_cols = st.columns(len(ids_now))
    remove_idx = None
    for i, rid in enumerate(ids_now):
        with chip_cols[i]:
            color = _color(i)
            st.markdown(
                f'<span style="background:{color};color:white;padding:3px 10px;'
                f'border-radius:12px;font-size:0.85em;font-weight:600">'
                f'R{i+1}: {rid[:10]}…</span>',
                unsafe_allow_html=True,
            )
            if st.button("Remove", key=f"icmp_rm_{i}"):
                remove_idx = i
    if remove_idx is not None:
        st.session_state["icmp_run_ids"].pop(remove_idx)
        st.session_state.pop("icmp_data", None)
        st.rerun()

if not st.session_state["icmp_run_ids"]:
    st.info(
        "Add at least two inference run IDs above, or select them from "
        "**Job Status → Compare Inferences**."
    )
    st.stop()

# =============================================================================
# Fetch
# =============================================================================

st.divider()
if st.button("Load / Refresh Comparison", type="primary"):
    fetched = {}
    for rid in st.session_state["icmp_run_ids"]:
        with st.spinner(f"Fetching {rid} …"):
            data = _fetch_inference(rid)
            if data is None:
                st.error(f"Could not load inference run `{rid}` — skipping.")
            elif data["status"].get("status") not in ("completed",):
                st.warning(
                    f"Run `{rid}` status is "
                    f"**{data['status'].get('status','?')}** — "
                    "only completed runs can be compared."
                )
            else:
                fetched[rid] = data
    st.session_state["icmp_data"] = fetched

icmp_data: dict = st.session_state.get("icmp_data", {})
if not icmp_data:
    st.info("Click **Load / Refresh Comparison** to fetch data.")
    st.stop()

run_ids_loaded = [r for r in st.session_state["icmp_run_ids"] if r in icmp_data]
if len(run_ids_loaded) < 1:
    st.error("No valid completed inference runs loaded.")
    st.stop()

# =============================================================================
# Task-type & source-run consistency checks
# =============================================================================

task_types   = []
source_runs  = []
for rid in run_ids_loaded:
    sd = icmp_data[rid]["status"]
    hp = sd.get("hyperparams", {})
    tt = hp.get("task_type") or sd.get("task_type") or "ppi"
    if tt == "inference":
        tt = "ppi"
    task_types.append(tt)
    source_runs.append(sd.get("source_run_id", ""))

unique_tasks   = list(dict.fromkeys(task_types))
unique_sources = list(dict.fromkeys(s for s in source_runs if s))

if len(unique_tasks) > 1:
    st.error(
        f"❌ **Mixed task types detected:** {', '.join(t.upper() for t in unique_tasks)}. "
        "Inference comparison requires all runs to be the same task type. "
        "Please remove the mismatched run(s)."
    )
    st.stop()

dominant_task = unique_tasks[0]
st.markdown(
    f"**Task type:** `{TASK_LABELS.get(dominant_task, dominant_task.upper())}` · "
    f"**Runs loaded:** {len(run_ids_loaded)}"
)

if len(unique_sources) > 1:
    st.warning(
        "⚠️ These inference runs were produced by **different training models** "
        f"({', '.join(s[:8] for s in unique_sources)}). "
        "Results are comparable only if all runs scored the same input pairs."
    )

# check if any run has labels (needed for ROC/PR/CM)
has_labels_any  = any(icmp_data[r]["metrics"].get("has_labels", False) for r in run_ids_loaded)
has_labels_all  = all(icmp_data[r]["metrics"].get("has_labels", False) for r in run_ids_loaded)

if has_labels_any and not has_labels_all:
    st.warning(
        "⚠️ Some runs were scored with ground-truth labels and some without. "
        "ROC / PR / Confusion Matrix will only show runs that have labels."
    )

source_details = {src: _fetch_training_detail(src) for src in unique_sources}
detail_models = []
for i, rid in enumerate(run_ids_loaded):
    src = icmp_data[rid]["status"].get("source_run_id", "")
    src_detail = source_details.get(src, {})
    src_hp = src_detail.get("hyperparams", {})
    if src_hp:
        detail_models.append({
            "label": _label(i, rid),
            "hyperparams": src_hp,
            "task_type": src_hp.get("task_type", dominant_task),
            "layer_configs": src_hp.get("layer_configs", []),
            "metrics": src_detail.get("metrics", {}),
        })

if detail_models:
    st.divider()
    st.subheader("Source Model Details")
    detail_tabs = st.tabs([model["label"] for model in detail_models])
    for tab, model in zip(detail_tabs, detail_models):
        with tab:
            render_model_details(
                st,
                pd,
                model["hyperparams"],
                model["task_type"],
                expanded=True,
                actual_params=model.get("metrics", {}).get("final", {}).get("trainable_params"),
            )

    st.divider()
    render_layer_difference_table(st, pd, detail_models)
else:
    st.info("Source model details are not available for the loaded inference runs.")

# =============================================================================
# Summary metric cards (per run)
# =============================================================================

st.divider()
metric_cols = st.columns(len(run_ids_loaded))
for i, rid in enumerate(run_ids_loaded):
    m   = icmp_data[rid]["metrics"]
    sd  = icmp_data[rid]["status"]
    col = _color(i)
    with metric_cols[i]:
        st.markdown(
            f'<div style="border-left:4px solid {col};padding:6px 10px;'
            f'background:{card_bg};border-radius:4px;margin-bottom:6px">'
            f'<b style="color:{col}">{_label(i, rid)}</b><br>'
            f'<small><code>{rid}</code></small></div>',
            unsafe_allow_html=True,
        )
        src = sd.get("source_run_id", "")
        if src:
            st.caption(f"Model: `{src[:12]}…`")
        n = len(m.get("probabilities", []))
        st.caption(f"Pairs scored: **{n:,}**")
        probs_arr = np.array(m.get("probabilities", []))
        if len(probs_arr):
            thr     = 0.5
            n_pos   = int((probs_arr >= thr).sum())
            n_neg   = len(probs_arr) - n_pos
            mean_p  = probs_arr.mean()
            st.caption(f"Predicted + : **{n_pos:,}**  − : **{n_neg:,}**")
            st.caption(f"Mean prob: **{mean_p:.3f}**")

# =============================================================================
# Summary metrics table (when labels present)
# =============================================================================

if has_labels_any:
    st.divider()
    st.subheader("Summary Metrics")

    rows = []
    for metric_key, metric_label in METRIC_ROWS:
        row = {"Metric": metric_label}
        best_val  = None
        best_runs = []
        for rid in run_ids_loaded:
            v = icmp_data[rid]["metrics"].get(metric_key)
            if v is not None:
                if best_val is None or v > best_val:
                    best_val  = v
                    best_runs = [rid]
                elif v == best_val:
                    best_runs.append(rid)
        for i, rid in enumerate(run_ids_loaded):
            v = icmp_data[rid]["metrics"].get(metric_key)
            cell = f"{v:.4f}" if v is not None else "—"
            if rid in best_runs and best_val is not None:
                cell = f"★ {cell}"
            row[_label(i, rid)] = cell
        rows.append(row)

    df_metrics = pd.DataFrame(rows).set_index("Metric")
    st.dataframe(df_metrics, use_container_width=True)
    st.caption("★ = best value for that metric across loaded runs.")

# =============================================================================
# Grouped bar chart — metrics
# =============================================================================

if has_labels_any:
    st.divider()
    st.subheader("Metrics Bar Chart")

    bar_metrics = [("accuracy", "Accuracy"), ("auroc", "AUROC"),
                   ("auprc", "AUPRC"), ("f1", "F1"), ("mcc", "MCC")]

    fig_bar = go.Figure()
    for i, rid in enumerate(run_ids_loaded):
        m = icmp_data[rid]["metrics"]
        values = [m.get(k) for k, _ in bar_metrics]
        if any(v is not None for v in values):
            fig_bar.add_trace(go.Bar(
                name=_label(i, rid),
                x=[lbl for _, lbl in bar_metrics],
                y=[v if v is not None else 0 for v in values],
                marker_color=_color(i),
                text=[f"{v:.3f}" if v is not None else "—" for v in values],
                textposition="outside",
            ))

    fig_bar.update_layout(
        barmode="group",
        height=400,
        template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(title="Run"),
        yaxis=dict(range=[0, 1.12], title="Score"),
        xaxis=dict(title="Metric"),
    )
    st.plotly_chart(fig_bar, use_container_width=True,
                    config={"toImageButtonOptions": {"format": "png", "filename": "metrics_bar_chart", "scale": 3}})

# =============================================================================
# ROC & PR Curves
# =============================================================================

label_runs = [r for r in run_ids_loaded if icmp_data[r]["metrics"].get("has_labels")]

if label_runs:
    st.divider()
    st.subheader("ROC & Precision–Recall Curves")

    fig_roc = go.Figure()
    fig_pr  = go.Figure()

    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        showlegend=False,
    ))

    has_roc = has_pr = False
    for i, rid in enumerate(run_ids_loaded):
        m      = icmp_data[rid]["metrics"]
        color  = _color(i)
        name   = _label(i, rid)
        probs  = np.array(m.get("probabilities", []))
        labels = np.array(m.get("labels", []))

        if not m.get("has_labels") or not len(labels) or not len(probs):
            continue

        try:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve
            fpr, tpr, _      = roc_curve(labels, probs)
            prec_a, rec_a, _ = precision_recall_curve(labels, probs)
            roc_auc_val      = auc(fpr, tpr)
            pr_auc_val       = auc(rec_a, prec_a)
        except Exception:
            continue

        fig_roc.add_trace(go.Scatter(
            x=fpr.tolist(), y=tpr.tolist(), mode="lines",
            name=f"{name} (AUROC={roc_auc_val:.3f})",
            line=dict(color=color, width=2),
        ))
        fig_pr.add_trace(go.Scatter(
            x=rec_a.tolist(), y=prec_a.tolist(), mode="lines",
            name=f"{name} (AUPRC={pr_auc_val:.3f})",
            line=dict(color=color, width=2),
        ))
        has_roc = has_pr = True

    _roc_pr_layout = dict(
        template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white",
        height=360,
        legend=dict(title="Run"),
    )
    _roc_pr_col, _ = st.columns([2, 1])

    if has_roc:
        st.markdown("**ROC Curves**")
        fig_roc.update_xaxes(title_text="False Positive Rate", range=[0, 1])
        fig_roc.update_yaxes(title_text="True Positive Rate",  range=[0, 1.02])
        fig_roc.update_layout(**_roc_pr_layout)
        with _roc_pr_col:
            st.plotly_chart(fig_roc, use_container_width=True,
                            config={"toImageButtonOptions": {"format": "png", "filename": "roc_curves", "scale": 3}})

    if has_pr:
        st.markdown("**Precision–Recall Curves**")
        fig_pr.update_xaxes(title_text="Recall",    range=[0, 1])
        fig_pr.update_yaxes(title_text="Precision", range=[0, 1.02])
        fig_pr.update_layout(**_roc_pr_layout)
        with _roc_pr_col:
            st.plotly_chart(fig_pr, use_container_width=True,
                            config={"toImageButtonOptions": {"format": "png", "filename": "pr_curves", "scale": 3}})

    if not has_roc and not has_pr:
        st.caption("ROC / PR data unavailable — run inference with labelled CSVs.")

# =============================================================================
# KDE probability distributions (overlaid)
# =============================================================================

st.divider()
st.subheader("Predicted Probability Distribution (KDE)")
st.caption(
    "Solid line = predicted probability density per run. "
    "When labels are available, dashed = negative class, solid = positive class."
)

thr_kde = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, key="icmp_kde_thr")

fig_kde = go.Figure()

for i, rid in enumerate(run_ids_loaded):
    m      = icmp_data[rid]["metrics"]
    probs  = np.array(m.get("probabilities", []))
    labels = np.array(m.get("labels", [])) if m.get("has_labels") else None
    color  = _color(i)
    name   = _label(i, rid)

    if not len(probs):
        continue

    if labels is not None and len(labels):
        pos_p = probs[labels == 1]
        neg_p = probs[labels == 0]
        if len(pos_p):
            xs, ys = _kde(pos_p)
            fig_kde.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=f"{name} (pos)",
                legendgroup=name,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.07)",
            ))
        if len(neg_p):
            xs, ys = _kde(neg_p)
            fig_kde.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=f"{name} (neg)",
                legendgroup=name, showlegend=False,
                line=dict(color=color, width=1.5, dash="dash"),
            ))
    else:
        xs, ys = _kde(probs)
        fig_kde.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=name,
            legendgroup=name,
            line=dict(color=color, width=2),
        ))

fig_kde.add_vline(
    x=thr_kde, line_dash="dash", line_color="#BA7517", line_width=2,
    annotation_text=f"thr={thr_kde:.2f}",
    annotation_position="top right",
    annotation_font_color="#BA7517",
)
fig_kde.update_xaxes(title_text="Predicted probability", range=[0, 1])
fig_kde.update_yaxes(title_text="Density")
fig_kde.update_layout(
    height=360, template="plotly_white",
    paper_bgcolor="white", plot_bgcolor="white",
    legend=dict(title="Run / class"),
)
st.plotly_chart(fig_kde, use_container_width=True,
                config={"toImageButtonOptions": {"format": "png", "filename": "probability_kde", "scale": 3}})

# =============================================================================
# Score scatter — all runs overlaid
# =============================================================================

st.divider()
st.subheader("Probability Score Scatter")
st.caption("Each point is one protein pair scored by that run. Hover for details.")

fig_sc = go.Figure()
for i, rid in enumerate(run_ids_loaded):
    m     = icmp_data[rid]["metrics"]
    probs = np.array(m.get("probabilities", []))
    if not len(probs):
        continue
    color = _color(i)
    name  = _label(i, rid)
    fig_sc.add_trace(go.Scatter(
        x=list(range(len(probs))),
        y=probs.tolist(),
        mode="markers",
        name=name,
        marker=dict(color=color, size=5, opacity=0.55),
    ))

thr_sc = st.slider("Highlight cut-off", 0.0, 1.0, 0.5, 0.01, key="icmp_sc_thr")
fig_sc.add_hline(y=thr_sc, line_dash="dash", line_color="#BA7517", line_width=1.5)
fig_sc.update_xaxes(title_text="Pair index")
fig_sc.update_yaxes(title_text="Predicted probability", range=[-0.02, 1.05])
fig_sc.update_layout(
    height=360, template="plotly_white",
    paper_bgcolor="white", plot_bgcolor="white",
    legend=dict(title="Run"),
)
st.plotly_chart(fig_sc, use_container_width=True,
                config={"toImageButtonOptions": {"format": "png", "filename": "probability_scatter", "scale": 3}})

# =============================================================================
# Confusion matrices (labelled runs only)
# =============================================================================

cm_runs = [r for r in run_ids_loaded if icmp_data[r]["metrics"].get("has_labels")]
if cm_runs:
    st.divider()
    st.subheader("Confusion Matrices")
    thr_cm = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01, key="icmp_cm_thr")

    cm_cols = st.columns(min(len(cm_runs), 5))
    for col_idx, rid in enumerate(cm_runs):
        i      = run_ids_loaded.index(rid)
        m      = icmp_data[rid]["metrics"]
        probs  = np.array(m.get("probabilities", []))
        labels = np.array(m.get("labels", []))
        color  = _color(i)

        if not len(probs) or not len(labels):
            continue

        preds = (probs >= thr_cm).astype(int)
        try:
            from sklearn.metrics import confusion_matrix as sk_cm
            cm_arr = sk_cm(labels, preds)
        except Exception:
            continue

        with cm_cols[col_idx % len(cm_cols)]:
            fig_cm, ax = plt.subplots(figsize=(3.2, 3.0))
            im = ax.imshow(cm_arr, cmap="Blues", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticklabels(["True 0", "True 1"])
            ax.set_title(_label(i, rid), color=color, fontweight="bold", fontsize=10)
            for r in range(cm_arr.shape[0]):
                for c in range(cm_arr.shape[1]):
                    v  = cm_arr[r, c]
                    tc = "white" if v > cm_arr.max() / 2 else "black"
                    ax.text(c, r, str(v), ha="center", va="center",
                            color=tc, fontsize=13, fontweight="bold")
            plt.tight_layout(pad=0.3)
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)

# =============================================================================
# Per-pair agreement table (top disagreements)
# =============================================================================

st.divider()
st.subheader("Per-Pair Score Comparison")
st.caption(
    "Pair-wise probabilities across all runs. Sorted by highest disagreement "
    "(standard deviation). Most useful when all runs scored the same input CSV."
)

# download results CSVs to get proteinA/B labels
results_dfs = {}
for rid in run_ids_loaded:
    try:
        rc = requests.get(f"{BACKEND}/download_results/{rid}", stream=True, timeout=10)
        if rc.ok:
            results_dfs[rid] = pd.read_csv(io.BytesIO(rc.content))
    except Exception:
        pass

# build merged dataframe keyed on pair index
if results_dfs:
    base_df = None
    for i, rid in enumerate(run_ids_loaded):
        df = results_dfs.get(rid)
        if df is None:
            continue
        df = df.copy()
        df[f"prob_{_label(i, rid)}"] = df.get("probability", pd.Series(dtype=float))
        df[f"pred_{_label(i, rid)}"] = (df.get("probability", pd.Series(dtype=float)) >= 0.5).astype(int)
        keep_cols = [c for c in ("proteinA", "proteinB", "drugA",
                                  "rnaA", "dnaA") if c in df.columns]
        keep_cols += [f"prob_{_label(i, rid)}", f"pred_{_label(i, rid)}"]
        if base_df is None:
            base_df = df[keep_cols].reset_index(drop=True)
        else:
            base_df = base_df.join(
                df[[f"prob_{_label(i, rid)}", f"pred_{_label(i, rid)}"]].reset_index(drop=True),
                how="outer",
            )

    if base_df is not None:
        prob_cols = [c for c in base_df.columns if c.startswith("prob_")]
        if len(prob_cols) > 1:
            base_df["disagreement_std"] = base_df[prob_cols].std(axis=1)
            base_df = base_df.sort_values("disagreement_std", ascending=False)

        n_show = st.slider("Rows to display", 10, min(500, len(base_df)), 50, 10,
                           key="icmp_table_rows")
        show_df = base_df.head(n_show).copy()

        # truncate long sequences
        for col in show_df.columns:
            if show_df[col].dtype == object:
                show_df[col] = show_df[col].astype(str).str[:30] + "…"

        # round probability columns
        for col in prob_cols:
            if col in show_df.columns:
                show_df[col] = show_df[col].round(4)

        st.dataframe(show_df, use_container_width=True, hide_index=True)
        st.caption(
            f"Showing top {n_show} of {len(base_df):,} pairs by disagreement. "
            "Download individual run CSVs from Job Status."
        )
    else:
        st.info("Could not download results CSVs — check backend connectivity.")
else:
    st.info("Could not retrieve results CSVs for pair-level comparison.")

# =============================================================================
# Threshold sensitivity table
# =============================================================================

st.divider()
st.subheader("Threshold Sensitivity")
st.caption("How many pairs each run predicts as positive at various thresholds.")

thrs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
thr_rows = []
for t in thrs:
    row = {"Threshold": t}
    for i, rid in enumerate(run_ids_loaded):
        probs = np.array(icmp_data[rid]["metrics"].get("probabilities", []))
        n_pos = int((probs >= t).sum()) if len(probs) else 0
        row[_label(i, rid)] = n_pos
    thr_rows.append(row)

st.dataframe(pd.DataFrame(thr_rows), use_container_width=True, hide_index=True)
