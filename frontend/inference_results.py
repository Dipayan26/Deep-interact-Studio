"""
Inference Results page — view any completed inference run by its run_id.
Reached from: Job Status → View (inference row), or Run Inference → View Results button.
"""

import io
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

from model_details import render_model_details

BACKEND = os.getenv("BACKEND_URL", "http://backend:8005")
is_dark = st.session_state.get("theme_mode", "Light") == "Dark"
plotly_template = st.session_state.get("plotly_template", "plotly_white")

C_POS   = "#185FA5"
C_NEG   = "#D85A30"
C_GREEN = "#1D9E75"
C_AMBER = "#BA7517"
BG      = "#343a42" if is_dark else "#ffffff"
TXT     = "#e6e8eb" if is_dark else "#1f2937"
SUBTXT  = "#c7ccd3" if is_dark else "#6b7280"

PLOTLY_LAYOUT = dict(
    template=plotly_template,
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="sans-serif", size=12, color=TXT),
    margin=dict(l=50, r=20, t=36, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color=TXT)),
    xaxis=dict(gridcolor="#545c68" if is_dark else "#e5e7eb", zerolinecolor="#545c68" if is_dark else "#e5e7eb"),
    yaxis=dict(gridcolor="#545c68" if is_dark else "#e5e7eb", zerolinecolor="#545c68" if is_dark else "#e5e7eb"),
)

# =============================================================================
# Header + Run ID selector
# =============================================================================

st.title("Inference Results")
st.markdown(
    "**Load a completed inference run to review predictions, metrics, score distributions, and result downloads. "
    "Inference run IDs are available from Run Inference or the Job Status table.**"
)
st.divider()

# Pre-populate from session state (set by job_status View button or inference.py)
_default_rid = st.session_state.get("infer_result_run_id", "")

ic1, ic2, ic3 = st.columns([5, 1.2, 1])
with ic1:
    _typed_rid = st.text_input(
        "Inference Run ID",
        value=_default_rid,
        placeholder="e.g. abc123ef",
        label_visibility="collapsed",
    ).strip()
with ic2:
    if st.button("Load", type="primary", use_container_width=True):
        st.session_state["infer_result_run_id"] = _typed_rid
        st.rerun()
with ic3:
    if st.button("Clear", use_container_width=True):
        st.session_state["infer_result_run_id"] = ""
        st.rerun()

rid = st.session_state.get("infer_result_run_id", "").strip()

if not rid:
    st.info("Enter an Inference Run ID above and click **Load** to view results.")
    st.stop()

# =============================================================================
# Poll status
# =============================================================================

try:
    sr = requests.get(f"{BACKEND}/check_status/{rid}", timeout=5)
    sd = sr.json()
except Exception as e:
    st.error(f"Could not reach backend: {e}")
    st.stop()

if "error" in sd:
    st.error(f"Run ID `{rid}` not found.")
    st.stop()

status = sd["status"]
colour = {"completed": "green", "running": "blue",
          "queued": "orange", "failed": "red"}.get(status, "gray")
st.markdown(f"**Status:** :{colour}[{status.capitalize()}]")

if status == "failed":
    st.error(f"Inference failed: {sd.get('result', 'check backend logs')}")
    st.stop()

if status in ("running", "queued"):
    st.info("Inference in progress…")
    col_auto = st.columns([1, 3])[0]
    auto = col_auto.checkbox("Auto-refresh", value=True, key="ir_auto")
    if auto:
        time.sleep(4)
        st.rerun()
    st.stop()

# =============================================================================
# Load inference job detail → source_run_id, task_type
# =============================================================================

try:
    jd_infer = requests.get(f"{BACKEND}/job_detail/{rid}", timeout=5).json()
except Exception:
    jd_infer = {}

source_run_id = jd_infer.get("source_run_id") or sd.get("source_run_id", "")
infer_hp      = jd_infer.get("hyperparams", {})
task_type     = infer_hp.get("task_type") or jd_infer.get("task_type", "ppi")
infer_label   = infer_hp.get("infer_label", "")

is_single = infer_hp.get("is_single", False)

# Header info bar
_label_part = f" · **{infer_label}**" if infer_label else ""
st.markdown(
    f"""
    <div style="display:inline-block; padding:6px 16px; border-radius:8px;
        background:#1a5fa520; border:1.5px solid #1a5fa560;
        font-size:0.92rem; font-weight:600; color:#1a5fa5; margin-bottom:4px;">
      {task_type.upper()}&nbsp;&nbsp;·&nbsp;&nbsp;
      Inference Run&nbsp;<code>{rid[:8]}…</code>&nbsp;&nbsp;·&nbsp;&nbsp;
      Source model&nbsp;<code>{source_run_id[:8] if source_run_id else '—'}…</code>
      {_label_part}
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Other inference runs from the same source model
# =============================================================================

if source_run_id:
    try:
        all_jobs = requests.get(f"{BACKEND}/jobs", timeout=5).json()
        sibling_runs = [
            j for j in all_jobs
            if j.get("job_type") == "inference"
            and j.get("source_run_id") == source_run_id
            and j.get("run_id") != rid
            and j.get("status") == "completed"
        ]
    except Exception:
        sibling_runs = []

    if sibling_runs:
        with st.expander(f"Other inference runs from model `{source_run_id[:8]}…` ({len(sibling_runs)} more)"):
            for sib in sibling_runs:
                sib_hp    = sib.get("hyperparams") or {}
                if isinstance(sib_hp, str):
                    import json as _json
                    try:
                        sib_hp = _json.loads(sib_hp)
                    except Exception:
                        sib_hp = {}
                sib_label = sib_hp.get("infer_label", "")
                sib_rid   = sib["run_id"]
                name_str  = f"`{sib_rid[:8]}…`" + (f" — {sib_label}" if sib_label else "")
                if st.button(f"View {name_str}", key=f"ir_sib_{sib_rid}"):
                    st.session_state["infer_result_run_id"] = sib_rid
                    st.rerun()

st.divider()

# =============================================================================
# Load all data sources
# =============================================================================

resp_csv = requests.get(f"{BACKEND}/download_results/{rid}", stream=True)
if resp_csv.status_code != 200:
    st.warning("Results file not available yet.")
    st.stop()
results_df = pd.read_csv(io.BytesIO(resp_csv.content))

try:
    mr          = requests.get(f"{BACKEND}/inference_metrics/{rid}", timeout=5)
    inf_metrics = mr.json() if mr.ok else {}
except Exception:
    inf_metrics = {}

has_labels = inf_metrics.get("has_labels", False)
probs      = np.array(inf_metrics.get("probabilities",
             results_df["probability"].tolist() if "probability" in results_df else []))
labels     = np.array(inf_metrics.get("labels", [])) if has_labels else None

try:
    tm        = requests.get(f"{BACKEND}/metrics/{source_run_id}", timeout=5)
    train_met = tm.json() if tm.ok else {}
except Exception:
    train_met = {}

history     = train_met.get("history", {})
has_history = bool(history.get("epoch"))

try:
    jd_src   = requests.get(f"{BACKEND}/job_detail/{source_run_id}", timeout=5).json()
except Exception:
    jd_src = {}

src_hp       = jd_src.get("hyperparams", {})
layer_configs = src_hp.get("layer_configs", [])
esm_dim       = int(src_hp.get("esm_dim", 480))

if src_hp:
    render_model_details(st, pd, src_hp, task_type, expanded=True)

# =============================================================================
# Summary metric cards
# =============================================================================

n_pairs    = len(results_df)
n_pos_pred = int((results_df.get("prediction", pd.Series(dtype=int)) == 1).sum())
n_neg_pred = n_pairs - n_pos_pred
mean_prob  = float(probs.mean()) if len(probs) else 0.0

if is_single and n_pairs == 1:
    prob_val = float(probs[0]) if len(probs) else mean_prob
    pred_val = int(results_df["prediction"].iloc[0]) if "prediction" in results_df.columns else int(prob_val >= 0.5)
    if task_type == "dtpi":
        label    = "Binding" if pred_val == 1 else "Non-Binding"
        prob_lbl = "Binding probability"
    else:
        label    = "Interacting" if pred_val == 1 else "Not Interacting"
        prob_lbl = "Interaction probability"
    card_col = C_POS if pred_val == 1 else C_NEG

    st.markdown(
        f"""
        <div style="border-radius:12px; padding:28px 36px; margin:12px 0;
            background:linear-gradient(135deg,{card_col}18,{card_col}08);
            border:2px solid {card_col}55; text-align:center;">
          <div style="font-size:2rem; font-weight:700; color:{card_col};">{label}</div>
          <div style="font-size:1.1rem; color:{SUBTXT}; margin-top:6px;">
            {prob_lbl}: <strong>{prob_val:.4f}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download result (.csv)",
        data=resp_csv.content,
        file_name=f"infer_result_{rid}.csv",
        mime="text/csv",
    )
else:
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Pairs scored",     f"{n_pairs:,}")
    mc2.metric("Predicted +",      f"{n_pos_pred:,}")
    mc3.metric("Predicted −",      f"{n_neg_pred:,}")
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
    file_name=f"infer_results_{rid}.csv",
    mime="text/csv",
)

st.divider()

# =============================================================================
# Dashboard tabs
# =============================================================================

def _kde(data: np.ndarray, bw: float = 0.04, n: int = 200):
    xs = np.linspace(0, 1, n)
    ys = np.array([
        np.mean(np.exp(-0.5 * ((x - data) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)))
        for x in xs
    ])
    return xs, ys


def _short_sequence(value, width: int = 32) -> str:
    text = str(value)
    return text if len(text) <= width else text[:width] + "..."


def _network_config(task: str) -> dict | None:
    configs = {
        "ppi": {
            "left_col": "proteinA",
            "right_col": "proteinB",
            "left_type": "Protein",
            "right_type": "Protein",
            "left_prefix": "P",
            "right_prefix": "P",
            "merge_sides": True,
            "uppercase": True,
        },
        "dtpi": {
            "left_col": "smiles",
            "right_col": "sequence",
            "left_type": "Drug",
            "right_type": "Protein",
            "left_prefix": "D",
            "right_prefix": "P",
            "merge_sides": False,
            "uppercase": False,
        },
        "rpi": {
            "left_col": "rna_sequence",
            "right_col": "protein_sequence",
            "left_type": "RNA",
            "right_type": "Protein",
            "left_prefix": "R",
            "right_prefix": "P",
            "merge_sides": False,
            "uppercase": True,
        },
        "pdi": {
            "left_col": "dna_sequence",
            "right_col": "protein_sequence",
            "left_type": "DNA",
            "right_type": "Protein",
            "left_prefix": "DNA",
            "right_prefix": "P",
            "merge_sides": False,
            "uppercase": True,
        },
    }
    return configs.get(task)


def _clean_node_value(value, uppercase: bool) -> str:
    text = str(value).strip()
    return text.upper() if uppercase else text


def _make_node_key(value: str, side: str, cfg: dict) -> tuple[str, str]:
    if cfg["merge_sides"]:
        return ("protein", value)
    return (side, value)


def _node_type(node_key: tuple[str, str], cfg: dict) -> str:
    if cfg["merge_sides"]:
        return cfg["left_type"]
    return cfg["left_type"] if node_key[0] == "left" else cfg["right_type"]


def _prepare_interaction_edges(
    df: pd.DataFrame,
    threshold: float,
    cfg: dict,
) -> tuple[list[dict], int]:
    left_col = cfg["left_col"]
    right_col = cfg["right_col"]
    required = {left_col, right_col, "probability"}
    if not required.issubset(df.columns):
        return [], 0

    work = df[[left_col, right_col, "probability"]].copy()
    work[left_col] = work[left_col].map(lambda value: _clean_node_value(value, cfg["uppercase"]))
    work[right_col] = work[right_col].map(lambda value: _clean_node_value(value, cfg["uppercase"]))
    work["probability"] = pd.to_numeric(work["probability"], errors="coerce")
    work = work.dropna(subset=["probability"])
    work = work[
        (work[left_col] != "")
        & (work[right_col] != "")
        & (work[left_col].str.lower() != "nan")
        & (work[right_col].str.lower() != "nan")
        & (work["probability"] >= threshold)
    ]

    self_loop_count = 0
    best_by_pair: dict[tuple[tuple[str, str], tuple[str, str]], dict] = {}
    for _, row in work.iterrows():
        left_value = row[left_col]
        right_value = row[right_col]
        left_key = _make_node_key(left_value, "left", cfg)
        right_key = _make_node_key(right_value, "right", cfg)
        if left_key == right_key:
            self_loop_count += 1
            continue

        prob = float(row["probability"])
        pair_key = tuple(sorted((left_key, right_key)))
        current = best_by_pair.get(pair_key)
        if current is None or prob > current["probability"]:
            best_by_pair[pair_key] = {"source": left_key, "target": right_key, "probability": prob}

    return list(best_by_pair.values()), self_loop_count


def _build_interaction_hub_payload(
    edges: list[dict],
    min_degree: int,
    top_hubs: int,
    max_partners: int,
    cfg: dict,
) -> dict:
    node_order: list[tuple[str, str]] = []
    seen_nodes: set[tuple[str, str]] = set()
    adjacency: dict[tuple[str, str], list[tuple[tuple[str, str], float]]] = {}

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        prob = float(edge["probability"])
        for node in (source, target):
            if node not in seen_nodes:
                seen_nodes.add(node)
                node_order.append(node)
            adjacency.setdefault(node, [])
        adjacency[source].append((target, prob))
        adjacency[target].append((source, prob))

    counts_by_prefix: dict[str, int] = {}
    node_ids = {}
    for node in node_order:
        prefix = cfg["left_prefix"] if cfg["merge_sides"] or node[0] == "left" else cfg["right_prefix"]
        counts_by_prefix[prefix] = counts_by_prefix.get(prefix, 0) + 1
        node_ids[node] = f"{prefix}{counts_by_prefix[prefix]}"

    degrees = {node: len(partners) for node, partners in adjacency.items()}
    weighted = {node: sum(prob for _, prob in partners) for node, partners in adjacency.items()}
    hubs = [
        node for node in node_order
        if degrees.get(node, 0) >= min_degree
    ]
    hubs.sort(key=lambda node: (-degrees[node], -weighted[node], node_ids[node]))
    selected_hubs = hubs[:top_hubs]

    displayed_edges: dict[tuple[tuple[str, str], tuple[str, str]], dict] = {}
    for hub in selected_hubs:
        partners = sorted(
            adjacency.get(hub, []),
            key=lambda item: (-item[1], node_ids.get(item[0], "")),
        )[:max_partners]
        for partner, prob in partners:
            key = tuple(sorted((hub, partner)))
            current = displayed_edges.get(key)
            if current is None or prob > current["probability"]:
                displayed_edges[key] = {"source": hub, "target": partner, "probability": prob}

    displayed_nodes = set(selected_hubs)
    for edge in displayed_edges.values():
        displayed_nodes.add(edge["source"])
        displayed_nodes.add(edge["target"])

    return {
        "node_ids": node_ids,
        "degrees": degrees,
        "weighted": weighted,
        "hubs": hubs,
        "selected_hubs": selected_hubs,
        "displayed_edges": list(displayed_edges.values()),
        "displayed_nodes": displayed_nodes,
        "node_order": node_order,
        "cfg": cfg,
    }


def _interaction_hub_positions(payload: dict) -> dict[tuple[str, str], tuple[float, float]]:
    selected_hubs = payload["selected_hubs"]
    displayed_nodes = payload["displayed_nodes"]
    displayed_edges = payload["displayed_edges"]
    node_ids = payload["node_ids"]

    positions: dict[str, tuple[float, float]] = {}
    n_hubs = len(selected_hubs)
    if n_hubs == 1:
        positions[selected_hubs[0]] = (0.0, 0.0)
    elif n_hubs > 1:
        radius = 1.0
        for idx, hub in enumerate(selected_hubs):
            angle = 2 * np.pi * idx / n_hubs
            positions[hub] = (radius * np.cos(angle), radius * np.sin(angle))

    hub_set = set(selected_hubs)
    neighbors_by_node: dict[str, list[tuple[str, float]]] = {}
    for edge in displayed_edges:
        source = edge["source"]
        target = edge["target"]
        prob = float(edge["probability"])
        if source in hub_set and target not in hub_set:
            neighbors_by_node.setdefault(target, []).append((source, prob))
        if target in hub_set and source not in hub_set:
            neighbors_by_node.setdefault(source, []).append((target, prob))

    for node in sorted(displayed_nodes - hub_set, key=lambda seq: node_ids[seq]):
        hub_links = sorted(neighbors_by_node.get(node, []), key=lambda item: (-item[1], node_ids[item[0]]))
        if len(hub_links) >= 2:
            xs = [positions[hub][0] for hub, _ in hub_links if hub in positions]
            ys = [positions[hub][1] for hub, _ in hub_links if hub in positions]
            if xs and ys:
                positions[node] = (float(np.mean(xs)) * 0.55, float(np.mean(ys)) * 0.55)
                continue
        anchor = hub_links[0][0] if hub_links else selected_hubs[0]
        ax, ay = positions.get(anchor, (0.0, 0.0))
        angle = np.arctan2(ay, ax)
        if ax == 0.0 and ay == 0.0:
            angle = 2 * np.pi * (len(positions) + 1) / max(len(displayed_nodes), 1)
        offset = 0.35 + 0.06 * (len(positions) % 5)
        positions[node] = (ax + offset * np.cos(angle), ay + offset * np.sin(angle))

    return positions


def _render_interaction_hub_network(payload: dict) -> go.Figure:
    positions = _interaction_hub_positions(payload)
    selected_hubs = set(payload["selected_hubs"])
    displayed_nodes = payload["displayed_nodes"]
    node_ids = payload["node_ids"]
    degrees = payload["degrees"]
    weighted = payload["weighted"]
    displayed_edges = payload["displayed_edges"]
    cfg = payload["cfg"]

    fig = go.Figure()
    if displayed_edges:
        min_prob = min(edge["probability"] for edge in displayed_edges)
        max_prob = max(edge["probability"] for edge in displayed_edges)
    else:
        min_prob = max_prob = 0.5

    for edge in displayed_edges:
        source = edge["source"]
        target = edge["target"]
        prob = float(edge["probability"])
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        scaled = 0.0 if max_prob == min_prob else (prob - min_prob) / (max_prob - min_prob)
        width = 1.4 + 4.0 * scaled
        alpha = 0.25 + 0.55 * scaled
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(width=width, color=f"rgba(24,95,165,{alpha:.3f})"),
            hoverinfo="text",
            text=(
                f"{node_ids[source]} - {node_ids[target]}<br>"
                f"Probability: {prob:.4f}"
            ),
            showlegend=False,
        ))

    node_rows = []
    type_colors = {
        "Protein": C_GREEN,
        "Drug": C_POS,
        "RNA": "#7C3AED",
        "DNA": "#0F766E",
    }
    for node in sorted(displayed_nodes, key=lambda item: (-int(item in selected_hubs), node_ids[item])):
        x, y = positions[node]
        degree = degrees.get(node, 0)
        value = node[1]
        mol_type = _node_type(node, cfg)
        node_rows.append({
            "node": node,
            "id": node_ids[node],
            "x": x,
            "y": y,
            "degree": degree,
            "weighted": weighted.get(node, 0.0),
            "is_hub": node in selected_hubs,
            "type": mol_type,
            "size": (18 if node in selected_hubs else 10) + min(degree, 30),
            "color": C_AMBER if node in selected_hubs else type_colors.get(mol_type, C_GREEN),
            "hover": (
                f"Node: {node_ids[node]}<br>"
                f"Type: {mol_type}<br>"
                f"Role: {'Hub' if node in selected_hubs else 'Partner'}<br>"
                f"Degree: {degree}<br>"
                f"Weighted degree: {weighted.get(node, 0.0):.4f}<br>"
                f"Length: {len(value)}<br>"
                f"Value: {_short_sequence(value, 60)}"
            ),
        })

    node_df = pd.DataFrame(node_rows)
    if not node_df.empty:
        for is_hub, label in [(False, "Partner nodes"), (True, "Hub nodes")]:
            group = node_df[node_df["is_hub"] == is_hub]
            if group.empty:
                continue
            fig.add_trace(go.Scatter(
                x=group["x"],
                y=group["y"],
                mode="markers+text",
                text=group["id"],
                textposition="top center",
                hovertext=group["hover"],
                hoverinfo="text",
                name=label,
                marker=dict(
                    size=group["size"],
                    color=group["color"],
                    line=dict(width=1.4, color=BG),
                    opacity=0.9,
                ),
            ))

    fig.update_layout({
        **PLOTLY_LAYOUT,
        "height": 620,
        "margin": dict(l=20, r=20, t=30, b=20),
        "xaxis": dict(visible=False),
        "yaxis": dict(visible=False),
        "showlegend": True,
    })
    return fig



# ---------------------------------------------------------------------------
# Section: ROC & PR Curves
# ---------------------------------------------------------------------------
if has_labels and labels is not None and len(labels):
    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve,
        confusion_matrix, matthews_corrcoef,
    )
    fpr, tpr, _      = roc_curve(labels, probs)
    roc_auc          = auc(fpr, tpr)
    prec_a, rec_a, _ = precision_recall_curve(labels, probs)
    pr_auc           = auc(rec_a, prec_a)

    st.markdown("##### ROC & Precision–Recall curves")
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("ROC curve", "Precision–Recall curve"))
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"AUROC = {roc_auc:.3f}",
        line=dict(color=C_POS, width=2),
        fill="tozeroy", fillcolor="rgba(24,95,165,0.09)"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color=C_NEG, dash="dash", width=1.5)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=rec_a, y=prec_a, mode="lines", name=f"AUPRC = {pr_auc:.3f}",
        line=dict(color=C_GREEN, width=2),
        fill="tozeroy", fillcolor="rgba(29,158,117,0.09)"
    ), row=1, col=2)
    fig.update_xaxes(title_text="False positive rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="True positive rate",  range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="Recall",    range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="Precision", range=[0, 1], row=1, col=2)
    fig.update_layout(**PLOTLY_LAYOUT, height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    # ---------------------------------------------------------------------------
    # Section: Confusion Matrix
    # ---------------------------------------------------------------------------
    st.markdown("##### Confusion matrix & per-class metrics")
    thr_cm = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01, key="ir_cm_thr")
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
    cm_fig.update_layout(**PLOTLY_LAYOUT, height=300)
    cm_fig.update_xaxes(side="top")
    cm_fig.update_yaxes(autorange="reversed")
    st.plotly_chart(cm_fig, use_container_width=True)

    prec_v = tp / (tp + fp) if (tp + fp) else 0
    rec_v  = tp / (tp + fn) if (tp + fn) else 0
    spec_v = tn / (tn + fp) if (tn + fp) else 0
    acc_v  = (tp + tn) / len(labels) if len(labels) else 0
    f1_v   = 2 * prec_v * rec_v / (prec_v + rec_v) if (prec_v + rec_v) else 0
    mcc_v  = matthews_corrcoef(labels, preds)
    st.dataframe(pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall / Sensitivity",
                   "Specificity", "F1", "MCC", "AUROC", "AUPRC"],
        "Value":  [f"{v:.4f}" for v in
                   [acc_v, prec_v, rec_v, spec_v, f1_v, mcc_v, roc_auc, pr_auc]],
    }), use_container_width=True, hide_index=True)

    st.divider()

# ---------------------------------------------------------------------------
# Section: Epoch Curves
# ---------------------------------------------------------------------------
st.markdown("##### Training epoch curves")
st.caption(
    f"From source training run `{source_run_id}`. "
    "Shows how loss and accuracy evolved during training."
)
if not has_history:
    st.info("No training history available for this model.")
else:
    epochs_h   = history.get("epoch", [])
    train_loss = [v for v in history.get("train_loss", []) if v is not None]
    val_loss   = [v for v in history.get("val_loss",   []) if v is not None]
    train_acc  = [v for v in history.get("train_acc",  []) if v is not None]
    val_acc    = [v for v in history.get("val_acc",    []) if v is not None]

    ep_loss = epochs_h[:len(train_loss)]
    ep_acc  = epochs_h[:len(train_acc)]

    fig_ep = make_subplots(rows=1, cols=2,
                           subplot_titles=("Loss per epoch", "Accuracy per epoch"))
    fig_ep.add_trace(go.Scatter(x=ep_loss, y=train_loss, mode="lines", name="Train loss",
        line=dict(color=C_POS, width=2)), row=1, col=1)
    fig_ep.add_trace(go.Scatter(x=ep_loss[:len(val_loss)], y=val_loss, mode="lines", name="Val loss",
        line=dict(color=C_NEG, width=2, dash="dot")), row=1, col=1)
    fig_ep.add_trace(go.Scatter(x=ep_acc, y=train_acc, mode="lines", name="Train acc",
        line=dict(color=C_POS, width=2), showlegend=False), row=1, col=2)
    fig_ep.add_trace(go.Scatter(x=ep_acc[:len(val_acc)], y=val_acc, mode="lines", name="Val acc",
        line=dict(color=C_GREEN, width=2, dash="dot"), showlegend=False), row=1, col=2)

    if train_met.get("early_stopped"):
        stopped_ep = train_met.get("epoch", ep_loss[-1] if ep_loss else None)
        if stopped_ep:
            for col in (1, 2):
                fig_ep.add_vline(x=stopped_ep, line_dash="dash", line_color=C_AMBER,
                    line_width=1.5, annotation_text="early stop",
                    annotation_font_color=C_AMBER,
                    annotation_position="top left", row=1, col=col)

    fig_ep.update_xaxes(title_text="Epoch")
    fig_ep.update_yaxes(title_text="Loss",     row=1, col=1)
    fig_ep.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig_ep.update_layout(**PLOTLY_LAYOUT, height=360)
    st.plotly_chart(fig_ep, use_container_width=True)

    if val_loss and val_acc:
        best_loss_ep = ep_loss[int(np.argmin(val_loss))] if val_loss else "—"
        best_acc_ep  = ep_acc[int(np.argmax(val_acc))]   if val_acc  else "—"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best val loss at epoch", str(best_loss_ep))
        c2.metric("Best val acc at epoch",  str(best_acc_ep))
        c3.metric("Final val loss", f"{val_loss[-1]:.4f}")
        c4.metric("Final val acc",  f"{val_acc[-1]:.4f}")

st.divider()
# ---------------------------------------------------------------------------
# Section: KDE
# ---------------------------------------------------------------------------
st.markdown("##### Kernel density estimate — predicted probabilities")
st.caption("KDE with bandwidth 0.04. When labels are supplied, classes are shown separately.")
thr_kde = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01, key="ir_kde_thr")
fig_kde = go.Figure()

if has_labels and labels is not None and len(labels):
    neg_p = probs[labels == 0]
    pos_p = probs[labels == 1]
    if len(neg_p):
        xs, ys = _kde(neg_p)
        fig_kde.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Negative (n={len(neg_p)})",
            line=dict(color=C_NEG, width=2), fill="tozeroy", fillcolor="rgba(216,90,48,0.12)"))
    if len(pos_p):
        xs, ys = _kde(pos_p)
        fig_kde.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Positive (n={len(pos_p)})",
            line=dict(color=C_POS, width=2), fill="tozeroy", fillcolor="rgba(24,95,165,0.12)"))
else:
    xs, ys = _kde(probs)
    fig_kde.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"All pairs (n={len(probs)})",
        line=dict(color=C_POS, width=2), fill="tozeroy", fillcolor="rgba(24,95,165,0.12)"))

fig_kde.add_vline(x=thr_kde, line_dash="dash", line_color=C_AMBER, line_width=2,
    annotation_text=f"thr = {thr_kde:.2f}", annotation_position="top right",
    annotation_font_color=C_AMBER)
fig_kde.update_xaxes(title_text="Predicted probability", range=[0, 1])
fig_kde.update_yaxes(title_text="Density")
fig_kde.update_layout(**PLOTLY_LAYOUT, height=360)
st.plotly_chart(fig_kde, use_container_width=True)

if has_labels and labels is not None and len(labels):
    neg_p = probs[labels == 0]
    pos_p = probs[labels == 1]
    if len(neg_p) and len(pos_p):
        xs_g = np.linspace(0, 1, 500)
        def _kdeval(data, bw=0.04):
            return np.array([
                np.mean(np.exp(-0.5 * ((x - data) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)))
                for x in xs_g
            ])
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        ovl = float(_trapz(np.minimum(_kdeval(neg_p), _kdeval(pos_p)), xs_g))
        st.caption(f"Overlap coefficient (OVL): **{ovl:.3f}** — lower means more separated distributions.")

st.divider()
# ---------------------------------------------------------------------------
# Section: SHAP
# ---------------------------------------------------------------------------
st.markdown("##### SHAP feature importance")
st.caption(
    "Mean absolute SHAP values via KernelExplainer. "
    "Supported for pooled PPI, DTPI, RPI, and PDI inference runs."
)

shap_data = st.session_state.get(f"shap_{rid}", None)
if shap_data is None:
    if st.button("Compute SHAP values", key="ir_shap_btn"):
        with st.spinner("Running KernelExplainer — this may take 30–90 s…"):
            try:
                sr = requests.get(f"{BACKEND}/shap/{rid}",
                    params={"n_background": 50, "n_explain": 100}, timeout=180)
                if sr.ok:
                    shap_data = sr.json()
                    if "error" in shap_data:
                        st.error(shap_data["error"])
                        shap_data = None
                    else:
                        st.session_state[f"shap_{rid}"] = shap_data
                        st.rerun()
                else:
                    try:
                        err = sr.json().get("error", "")
                    except Exception:
                        err = sr.text
                    suffix = f": {err}" if err else ""
                    st.error(f"SHAP endpoint returned {sr.status_code}{suffix}")
            except requests.exceptions.Timeout:
                st.error("SHAP computation timed out.")
            except Exception as e:
                st.error(f"SHAP request failed: {e}")
    else:
        st.info("Click **Compute SHAP values** to run KernelExplainer on the model.")

if shap_data is not None:
    split_d    = int(shap_data.get("esm_dim", shap_data.get("left_dim", 480)))
    left_dim   = int(shap_data.get("left_dim", split_d))
    right_dim  = int(shap_data.get("right_dim", max(0, len(shap_data.get("all_dims", [])) - split_d)))
    left_label = shap_data.get("left_label", "Protein A")
    right_label = shap_data.get("right_label", "Protein B")
    global_top = shap_data.get("global_top", [])
    eA_top     = shap_data.get("eA_top", [])
    eB_top     = shap_data.get("eB_top", [])
    eA_mean    = shap_data.get("eA_mean", 0)
    eB_mean    = shap_data.get("eB_mean", 0)

    grp_fig = go.Figure(go.Bar(
        x=[left_label, right_label],
        y=[eA_mean, eB_mean],
        marker_color=[C_POS, C_GREEN],
        text=[f"{eA_mean:.4f}", f"{eB_mean:.4f}"],
        textposition="outside",
    ))
    grp_fig.update_layout(**PLOTLY_LAYOUT, height=260,
        yaxis_title="Mean |SHAP|", title_text="Feature group importance")
    st.plotly_chart(grp_fig, use_container_width=True)

    def _global_dim_label(dim):
        if dim < left_dim:
            return f"dim {dim} ({left_label})"
        if dim < left_dim + right_dim:
            return f"dim {dim - left_dim} ({right_label})"
        return f"dim {dim} (interaction)"

    st.markdown("**Top 15 dimensions by |SHAP| — global**")
    if global_top:
        dims   = [_global_dim_label(d["dim"]) for d in global_top]
        values = [d["value"] for d in global_top]
        colors = [C_POS if d["dim"] < left_dim else C_GREEN if d["dim"] < left_dim + right_dim else C_AMBER for d in global_top]
        top_fig = go.Figure(go.Bar(
            x=values[::-1], y=dims[::-1], orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.4f}" for v in values[::-1]], textposition="outside",
        ))
        top_layout = {
            **PLOTLY_LAYOUT,
            "height": max(300, len(global_top) * 26),
            "margin": dict(l=130, r=60, t=20, b=40),
        }
        top_fig.update_layout(**top_layout)
        st.plotly_chart(top_fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Top 10 — {left_label} dimensions**")
        if eA_top:
            st.dataframe(pd.DataFrame([
                {f"Dim ({left_label})": d["dim"], "Mean |SHAP|": round(d["value"], 6)}
                for d in eA_top[:10]
            ]), use_container_width=True, hide_index=True)
    with col_b:
        st.markdown(f"**Top 10 — {right_label} dimensions**")
        if eB_top:
            st.dataframe(pd.DataFrame([
                {f"Dim ({right_label})": d["dim"], "Mean |SHAP|": round(d["value"], 6)}
                for d in eB_top[:10]
            ]), use_container_width=True, hide_index=True)

    all_dims = shap_data.get("all_dims", [])
    if all_dims:
        st.markdown("**Full SHAP spectrum across all embedding dimensions**")
        full_x = list(range(len(all_dims)))
        spec_fig = go.Figure()
        spec_fig.add_trace(go.Bar(x=full_x[:left_dim], y=all_dims[:left_dim],
            name=f"{left_label} dims", marker_color=C_POS, opacity=0.7))
        spec_fig.add_trace(go.Bar(x=full_x[left_dim:left_dim + right_dim], y=all_dims[left_dim:left_dim + right_dim],
            name=f"{right_label} dims", marker_color=C_GREEN, opacity=0.7))
        if len(all_dims) > left_dim + right_dim:
            spec_fig.add_trace(go.Bar(x=full_x[left_dim + right_dim:], y=all_dims[left_dim + right_dim:],
                name="Interaction dims", marker_color=C_AMBER, opacity=0.55))
        spec_fig.add_vline(x=left_dim - 0.5, line_dash="dash", line_color="#888780",
            line_width=1.5, annotation_text=f"{left_label} | {right_label}", annotation_font_color="#888780")
        spec_fig.update_layout(**PLOTLY_LAYOUT, height=280,
            xaxis_title="Embedding dimension", yaxis_title="Mean |SHAP|", barmode="overlay")
        st.plotly_chart(spec_fig, use_container_width=True)

st.divider()
# ---------------------------------------------------------------------------
# Section: Probability Distribution
# ---------------------------------------------------------------------------
st.markdown("##### Predicted probability distribution")
if len(probs) == 0:
    st.info("No probability data available.")
else:
    counts, bin_edges = np.histogram(probs, bins=25, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    thr_hist    = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01, key="ir_hist_thr")
    colors      = [C_NEG if c >= thr_hist else C_POS for c in bin_centers]
    hist_fig    = go.Figure(go.Bar(
        x=bin_centers, y=counts, marker_color=colors,
        width=(bin_edges[1] - bin_edges[0]) * 0.92,
    ))
    hist_fig.add_vline(x=thr_hist, line_dash="dash", line_color=C_AMBER, line_width=2,
        annotation_text=f"thr = {thr_hist:.2f}", annotation_position="top right",
        annotation_font_color=C_AMBER)
    hist_fig.update_xaxes(title_text="Predicted probability", range=[0, 1])
    hist_fig.update_yaxes(title_text="Count")
    hist_fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
    st.plotly_chart(hist_fig, use_container_width=True)
    n_pos_thr = int((probs >= thr_hist).sum())
    st.caption(
        f"{n_pos_thr} / {len(probs)} pairs predicted positive at threshold {thr_hist:.2f}  |  "
        f"Mean: {probs.mean():.3f}  ·  Median: {np.median(probs):.3f}  ·  Std: {probs.std():.3f}"
    )

st.divider()
# ---------------------------------------------------------------------------
# Section: Score Scatter
# ---------------------------------------------------------------------------
st.markdown("##### Probability scatter — pair index vs. score")
scatter_df = results_df.copy()
scatter_df["idx"] = range(len(scatter_df))
if "probability" not in scatter_df.columns:
    scatter_df["probability"] = probs
if task_type == "dtpi":
    col_a_name, col_b_name = "smiles", "sequence"
elif task_type == "rpi":
    col_a_name, col_b_name = "rna_sequence", "protein_sequence"
elif task_type == "pdi":
    col_a_name, col_b_name = "dna_sequence", "protein_sequence"
else:
    col_a_name, col_b_name = "proteinA", "proteinB"
short_a = scatter_df.get(col_a_name, scatter_df.iloc[:, 0]).astype(str).str[:20] + "…"
short_b = scatter_df.get(col_b_name, scatter_df.iloc[:, 1]).astype(str).str[:20] + "…"
scatter_df["hover"] = short_a + " × " + short_b
if has_labels and labels is not None and len(labels) == len(scatter_df):
    scatter_df["true_label"] = labels.astype(int).astype(str)
    fig_sc = px.scatter(scatter_df, x="idx", y="probability",
        color="probability", color_continuous_scale=[C_NEG, C_POS],
        symbol="true_label", symbol_map={"0": "circle-open", "1": "circle"},
        hover_name="hover",
        hover_data={"idx": False, "probability": ":.3f", "true_label": True},
        labels={"idx": "Pair index", "probability": "Probability", "true_label": "True label"})
else:
    fig_sc = px.scatter(scatter_df, x="idx", y="probability",
        color="probability", color_continuous_scale=[C_NEG, C_POS],
        hover_name="hover",
        hover_data={"idx": False, "probability": ":.3f"},
        labels={"idx": "Pair index", "probability": "Probability"})
thr_sc = st.slider("Highlight cut-off", 0.0, 1.0, 0.5, 0.01, key="ir_sc_thr")
fig_sc.add_hline(y=thr_sc, line_dash="dash", line_color=C_AMBER, line_width=1.5)
fig_sc.update_traces(marker=dict(size=7, opacity=0.75))
fig_sc.update_layout(**PLOTLY_LAYOUT, height=360,
                     coloraxis_colorbar=dict(title="P(interact)"))
st.plotly_chart(fig_sc, use_container_width=True)
n_above = int((scatter_df["probability"] >= thr_sc).sum())
st.caption(f"{n_above} / {n_pairs} pairs ≥ {thr_sc:.2f}")

st.divider()
# ---------------------------------------------------------------------------
# Section: Raw Results
# ---------------------------------------------------------------------------
st.markdown("##### All scored pairs")
if task_type == "dtpi":
    search_cols = ["smiles", "sequence"]
    search_ph   = "e.g. CC(=O)… or MKTAY…"
elif task_type == "rpi":
    search_cols = ["rna_sequence", "protein_sequence"]
    search_ph   = "e.g. AUGCUU… or MKTAY…"
elif task_type == "pdi":
    search_cols = ["dna_sequence", "protein_sequence"]
    search_ph   = "e.g. ATGCTT… or MKTAY…"
else:
    search_cols = ["proteinA", "proteinB"]
    search_ph   = "e.g. MKTAY…"
search  = st.text_input("Filter by substring", placeholder=search_ph, key="ir_raw_search")
show_df = results_df.copy()
if search.strip():
    mask = pd.Series([False] * len(show_df), index=show_df.index)
    for sc in search_cols:
        if sc in show_df.columns:
            mask |= show_df[sc].astype(str).str.contains(search, case=False, na=False)
    show_df = show_df[mask]
for col in search_cols:
    if col in show_df.columns:
        show_df[col] = show_df[col].astype(str).str[:40] + "…"
st.dataframe(show_df, use_container_width=True, hide_index=True)
st.caption(f"Showing {len(show_df):,} of {n_pairs:,} rows")

st.divider()
st.markdown("##### Threshold sensitivity")
thr_rows = []
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    n = int((results_df.get("probability", pd.Series(probs)) >= t).sum())
    thr_rows.append({"Threshold": t, "Predicted positive": n, "Predicted negative": n_pairs - n})
st.dataframe(pd.DataFrame(thr_rows), use_container_width=True, hide_index=True)

st.divider()
st.markdown("##### Interaction hub network")

network_cfg = _network_config(task_type)
if network_cfg is None:
    st.info("Interaction hub network visualization is available for PPI, DTPI, RPI, and PDI inference runs.")
elif not {network_cfg["left_col"], network_cfg["right_col"], "probability"}.issubset(results_df.columns):
    st.warning(
        "Interaction network requires "
        f"{network_cfg['left_col']}, {network_cfg['right_col']}, and probability columns in the results CSV."
    )
else:
    net_c1, net_c2, net_c3, net_c4 = st.columns(4)
    with net_c1:
        net_threshold = st.slider(
            "Network threshold",
            0.0, 1.0, 0.50, 0.01,
            key=f"ir_{task_type}_net_thr",
        )

    interaction_edges, self_loop_count = _prepare_interaction_edges(results_df, net_threshold, network_cfg)

    if not interaction_edges:
        st.info("No positive interactions at this threshold.")
        if self_loop_count:
            st.caption(f"{self_loop_count:,} self-interaction row(s) were not drawn as network edges.")
    else:
        base_payload = _build_interaction_hub_payload(
            interaction_edges,
            min_degree=1,
            top_hubs=len(interaction_edges) * 2,
            max_partners=len(interaction_edges) * 2,
            cfg=network_cfg,
        )
        degrees = base_payload["degrees"]
        max_degree = max(degrees.values()) if degrees else 1

        if max_degree < 2:
            with net_c2:
                st.metric("Maximum degree", str(max_degree))
            ms1, ms2, ms3, ms4 = st.columns(4)
            ms1.metric("Positive edges", f"{len(interaction_edges):,}")
            ms2.metric("Unique nodes", f"{len(base_payload['node_ids']):,}")
            ms3.metric("Hubs found", "0")
            ms4.metric("Displayed graph", "0 nodes / 0 edges")
            st.info("No hub nodes found; every node has only one positive interaction at this threshold.")
            if self_loop_count:
                st.caption(f"{self_loop_count:,} self-interaction row(s) were not drawn as network edges.")
        else:
            min_degree_key = f"ir_{task_type}_net_min_degree"
            if st.session_state.get(min_degree_key, 2) > int(max_degree):
                st.session_state[min_degree_key] = int(max_degree)
            if st.session_state.get(min_degree_key, 2) < 2:
                st.session_state[min_degree_key] = 2
            with net_c2:
                if int(max_degree) == 2:
                    min_hub_degree = 2
                    st.metric("Minimum hub degree", "2")
                    st.caption("Degree 2 is the only hub threshold available for this run.")
                else:
                    min_hub_degree = st.slider(
                        "Minimum hub degree",
                        2, int(max_degree),
                        2,
                        1,
                        key=min_degree_key,
                    )

            hub_candidates = [
                seq for seq, degree in degrees.items()
                if degree >= min_hub_degree
            ]
            hub_candidates.sort(
                key=lambda seq: (
                    -degrees[seq],
                    -base_payload["weighted"].get(seq, 0.0),
                    base_payload["node_ids"][seq],
                )
            )

            if not hub_candidates:
                ms1, ms2, ms3, ms4 = st.columns(4)
                ms1.metric("Positive edges", f"{len(interaction_edges):,}")
                ms2.metric("Unique nodes", f"{len(base_payload['node_ids']):,}")
                ms3.metric("Hubs found", "0")
                ms4.metric("Displayed graph", "0 nodes / 0 edges")
                st.info("No hub nodes found at the selected minimum degree.")
                st.caption("All positive interactions are isolated or below the selected hub degree.")
            else:
                max_hubs_available = len(hub_candidates)
                top_hubs_key = f"ir_{task_type}_net_top_hubs"
                if st.session_state.get(top_hubs_key, 1) > max_hubs_available:
                    st.session_state[top_hubs_key] = max_hubs_available
                with net_c3:
                    if max_hubs_available == 1:
                        top_hubs = 1
                        st.metric("Top hubs", "1")
                        st.caption("Only one hub matches the selected degree.")
                    else:
                        top_hubs = st.slider(
                            "Top hubs",
                            1, max_hubs_available,
                            min(10, max_hubs_available),
                            1,
                            key=top_hubs_key,
                        )
                max_neighbors_available = max(
                    max((degrees.get(seq, 0) for seq in hub_candidates[:top_hubs]), default=1),
                    1,
                )
                partners_key = f"ir_{task_type}_net_max_partners"
                if st.session_state.get(partners_key, 1) > max_neighbors_available:
                    st.session_state[partners_key] = max_neighbors_available
                with net_c4:
                    if max_neighbors_available == 1:
                        max_partners = 1
                        st.metric("Partners per hub", "1")
                        st.caption("Only one partner is available for the selected hub.")
                    else:
                        max_partners = st.slider(
                            "Partners per hub",
                            1, max_neighbors_available,
                            min(25, max_neighbors_available),
                            1,
                            key=partners_key,
                        )

                payload = _build_interaction_hub_payload(
                    interaction_edges,
                    min_degree=min_hub_degree,
                    top_hubs=top_hubs,
                    max_partners=max_partners,
                    cfg=network_cfg,
                )
                displayed_edges = payload["displayed_edges"]
                displayed_nodes = payload["displayed_nodes"]

                ms1, ms2, ms3, ms4 = st.columns(4)
                ms1.metric("Positive edges", f"{len(interaction_edges):,}")
                ms2.metric("Unique nodes", f"{len(base_payload['node_ids']):,}")
                ms3.metric("Hubs found", f"{len(payload['hubs']):,}")
                ms4.metric("Displayed graph", f"{len(displayed_nodes):,} nodes / {len(displayed_edges):,} edges")

                if self_loop_count:
                    st.caption(f"{self_loop_count:,} self-interaction row(s) were not drawn as network edges.")

                if not displayed_edges:
                    st.info("No hub-neighbor edges are available with the current controls.")
                else:
                    fig_net = _render_interaction_hub_network(payload)
                    st.plotly_chart(fig_net, use_container_width=True)

                    hub_rows = []
                    for hub in payload["selected_hubs"]:
                        displayed_partner_count = sum(
                            1 for edge in displayed_edges
                            if edge["source"] == hub or edge["target"] == hub
                        )
                        hub_rows.append({
                            "Hub ID": payload["node_ids"][hub],
                            "Type": _node_type(hub, network_cfg),
                            "Degree": payload["degrees"].get(hub, 0),
                            "Weighted degree": round(payload["weighted"].get(hub, 0.0), 4),
                            "Displayed partners": displayed_partner_count,
                            "Length": len(hub[1]),
                            "Value preview": _short_sequence(hub[1], 48),
                        })

                    st.markdown("**Selected hubs**")
                    st.dataframe(pd.DataFrame(hub_rows), use_container_width=True, hide_index=True)

                    selected_hub_set = set(payload["selected_hubs"])
                    mapping_rows = []
                    for seq in sorted(displayed_nodes, key=lambda value: payload["node_ids"][value]):
                        mapping_rows.append({
                            "Node ID": payload["node_ids"][seq],
                            "Type": _node_type(seq, network_cfg),
                            "Role": "Hub" if seq in selected_hub_set else "Partner",
                            "Degree": payload["degrees"].get(seq, 0),
                            "Weighted degree": round(payload["weighted"].get(seq, 0.0), 4),
                            "Length": len(seq[1]),
                            "Value": seq[1],
                        })

                    mapping_df = pd.DataFrame(mapping_rows)
                    st.markdown("**Displayed node mapping**")
                    st.dataframe(mapping_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download displayed node mapping (.csv)",
                        data=mapping_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{task_type}_hub_nodes_{rid}.csv",
                        mime="text/csv",
                    )
