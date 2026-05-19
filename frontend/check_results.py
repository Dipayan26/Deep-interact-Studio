import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from model_details import render_model_details

BACKEND = os.getenv("BACKEND_URL", "http://backend:8005")
plotly_template = st.session_state.get("plotly_template", "plotly_white")
EXAMPLE_RUNS = [
    ("Example PPI", "example_ppi", True),
    ("Example DTPI", "example_dtpi", True),
    ("Example RPI", "example_rpi", True),
    ("Example PDI", "example_pdi", False),
]

st.title("Check Model Results")
st.markdown(
    "**Enter a training run ID to inspect status, metrics, curves, dataset details, and downloadable model artefacts. "
    "Use Job Status to find completed training run IDs quickly.**"
)

st.divider()

default_id    = st.session_state.get("last_run_id", "")
default_token = st.session_state.get("last_cancel_token", "")

run_id = st.text_input("Run ID", value=default_id, placeholder="e.g. 3f2a1c8b")

col_check, col_auto, col_reset = st.columns([1, 2, 1])
with col_check:
    check_btn = st.button("Check Status", type="primary")
with col_auto:
    auto_refresh = st.checkbox("Auto-refresh while training", value=False)
with col_reset:
    if st.button("Reset", type="secondary"):
        st.session_state.pop("last_run_id", None)
        st.session_state.pop("last_cancel_token", None)
        st.session_state.pop("active_rid", None)
        st.session_state.pop("active_is_example", None)
        st.rerun()

st.markdown("**Example runs**")
example_cols = st.columns(4)
for idx, (label, example_id, available) in enumerate(EXAMPLE_RUNS):
    with example_cols[idx]:
        if st.button(label, disabled=not available, use_container_width=True, key=f"load_{example_id}"):
            st.session_state["active_rid"] = example_id
            st.session_state["active_is_example"] = True
            st.rerun()
if not EXAMPLE_RUNS[-1][2]:
    st.caption("PDI example coming soon.")

rid = (run_id or "").strip()

if check_btn and rid:
    st.session_state["active_rid"] = rid
    st.session_state["active_is_example"] = False

active_rid = st.session_state.get("active_rid", "")
active_is_example = bool(st.session_state.get("active_is_example", False))

if not (check_btn or auto_refresh or active_rid):
    st.stop()

if active_is_example:
    rid = active_rid
elif not rid:
    rid = active_rid

if not rid:
    st.error("Enter a run ID.")
    st.stop()

def _endpoint(kind: str) -> str:
    if active_is_example:
        return f"{BACKEND}/example_runs/{rid}/{kind}"
    return f"{BACKEND}/{kind}/{rid}"


try:
    status_r    = requests.get(_endpoint("check_status"), timeout=5)
    status_data = status_r.json()
except Exception as e:
    st.error(f"Could not reach backend: {e}")
    st.stop()

try:
    metrics_r    = requests.get(_endpoint("metrics"), timeout=5)
    metrics_data = metrics_r.json() if metrics_r.ok else {}
except Exception:
    metrics_data = {}

if "error" in status_data:
    st.error("Run ID not found.")
    st.stop()

status = status_data["status"]
hp     = status_data.get("hyperparams", {})
task_type = hp.get("task_type", status_data.get("job_type", "ppi"))

status_colours = {
    "completed": "green",
    "running":   "blue",
    "queued":    "orange",
    "failed":    "red",
    "cancelled": "gray",
}

# ── Status badge line ──────────────────────────────────────────────────────────
final_now = metrics_data.get("final", {})
badge_parts = [task_type.upper()]
if final_now.get("val_acc") is not None:
    badge_parts.append(f"acc = {final_now['val_acc']:.2f}")
if final_now.get("auroc") is not None:
    badge_parts.append(f"auroc = {final_now['auroc']:.2f}")
badge_str = "  ·  ".join(badge_parts)

st.markdown(
    f"**Status:** :{status_colours.get(status, 'gray')}[{status.capitalize()}]"
    f"   `{badge_str}`"
)
if active_is_example:
    title = status_data.get("title") or "Example Run"
    source = status_data.get("source_run_id")
    source_text = f" · source `{source}`" if source else ""
    st.info(f"{title} · permanent demo result{source_text}")

if hp:
    render_model_details(
        st,
        pd,
        hp,
        task_type,
        expanded=True,
        actual_params=final_now.get("trainable_params"),
    )

# ── progress bar ──────────────────────────────────────────────────────────────
epoch        = metrics_data.get("epoch", 0)
total_epochs = metrics_data.get("total_epochs", 0)
stage        = metrics_data.get("stage")
stage_msg    = metrics_data.get("message") or ""
stage_prog   = float(metrics_data.get("progress") or 0.0)
stage_cur    = int(metrics_data.get("current") or 0)
stage_total  = int(metrics_data.get("total") or 0)
if total_epochs > 0:
    label = f"Epoch {epoch} / {total_epochs}"
    if metrics_data.get("early_stopped"):
        label += "  (early stopped)"
    st.progress(epoch / total_epochs, text=label)
elif status in {"queued", "running"}:
    if stage:
        stage_label = stage.replace("_", " ").capitalize()
        count_label = f"  ({stage_cur:,}/{stage_total:,})" if stage_total else ""
        text = f"{stage_label}: {stage_msg or 'Working...'}{count_label}"
        st.progress(max(0.0, min(1.0, stage_prog)), text=text)
    else:
        st.progress(0.0, text="Waiting for worker to start...")

# ── training curves ───────────────────────────────────────────────────────────
history = metrics_data.get("history", {})
if history.get("epoch"):
    c1, c2 = st.columns(2)
    with c1:
        loss_df = pd.DataFrame({
            "train": history.get("train_loss", []),
            "val":   history.get("val_loss",   []),
        }, index=history["epoch"])
        st.caption("Loss")
        st.line_chart(loss_df, x_label="Epoch", y_label="Loss")
    with c2:
        acc_df = pd.DataFrame({
            "train": history.get("train_acc", []),
            "val":   history.get("val_acc",   []),
        }, index=history["epoch"])
        st.caption("Accuracy")
        st.line_chart(acc_df, x_label="Epoch", y_label="Accuracy")

# ── final metrics ─────────────────────────────────────────────────────────────
if status == "completed":
    final = metrics_data.get("final", {})
    if final:
        st.divider()
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Val Accuracy", f"{final['val_acc']:.4f}"   if final.get("val_acc")   is not None else "—")
        m2.metric("AUROC",        f"{final['auroc']:.4f}"     if final.get("auroc")     is not None else "—")
        m3.metric("Avg Precision",f"{final['ap']:.4f}"        if final.get("ap")        is not None else "—")
        m4.metric("Precision",    f"{final['precision']:.4f}" if final.get("precision") is not None else "—")
        m5.metric("Recall",       f"{final['recall']:.4f}"    if final.get("recall")    is not None else "—")
        m6.metric("F1",           f"{final['f1']:.4f}"        if final.get("f1")        is not None else "—")

    # ── Diagnostics plots ─────────────────────────────────────────────────────
    cm_data   = metrics_data.get("confusion_matrix")
    roc_data  = metrics_data.get("roc_curve")
    pr_data   = metrics_data.get("pr_curve")
    hist_data = metrics_data.get("prob_hist")

    val_probs  = metrics_data.get("val_probs")
    val_labels = metrics_data.get("val_labels")
    has_live_cm = val_probs and val_labels and len(val_probs) == len(val_labels)

    has_diag = any(x is not None for x in [cm_data, roc_data, pr_data, hist_data])
    if has_diag:
        st.divider()
        st.subheader("Diagnostics")

        if has_live_cm:
            threshold = st.slider(
                "Decision threshold", min_value=0.0, max_value=1.0,
                value=0.5, step=0.01, key="cr_threshold",
            )
            vp  = np.array(val_probs, dtype=float)
            vl  = np.array(val_labels, dtype=int)
            prd = (vp >= threshold).astype(int)
            tn  = int(((prd == 0) & (vl == 0)).sum())
            fp  = int(((prd == 1) & (vl == 0)).sum())
            fn  = int(((prd == 0) & (vl == 1)).sum())
            tp  = int(((prd == 1) & (vl == 1)).sum())
            cm_data = [[tn, fp], [fn, tp]]
        else:
            threshold = 0.5

        dc1, dc2, dc3, dc4 = st.columns(4)

        # 1. Confusion Matrix
        with dc1:
            if cm_data is not None:
                try:
                    cm_arr = np.array(cm_data, dtype=int)   # [[TN,FP],[FN,TP]]
                    fig_cm, ax_cm = plt.subplots(figsize=(3, 2.8))
                    im = ax_cm.imshow(cm_arr, cmap="Blues", aspect="auto")
                    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
                    ax_cm.set_xticks([0, 1])
                    ax_cm.set_yticks([0, 1])
                    ax_cm.set_xticklabels(["Pred 0", "Pred 1"])
                    ax_cm.set_yticklabels(["True 0", "True 1"])
                    ax_cm.set_title(f"Confusion Matrix (t={threshold:.2f})")
                    for r in range(2):
                        for c in range(2):
                            val = cm_arr[r, c]
                            color = "white" if cm_arr[r, c] > cm_arr.max() / 2 else "black"
                            ax_cm.text(c, r, str(val), ha="center", va="center",
                                       color=color, fontsize=14, fontweight="bold")
                    plt.tight_layout(pad=0.3)
                    st.pyplot(fig_cm, use_container_width=True)
                    plt.close(fig_cm)
                except Exception:
                    st.caption("Confusion matrix unavailable.")
            else:
                st.caption("No confusion matrix data.")

        # 2. ROC Curve
        with dc2:
            if roc_data is not None:
                try:
                    fpr = [v for v in roc_data.get("fpr", []) if v is not None]
                    tpr = [v for v in roc_data.get("tpr", []) if v is not None]
                    auroc_val = final.get("auroc") if final else None
                    title_str = f"ROC Curve  (AUROC={auroc_val:.4f})" if auroc_val is not None else "ROC Curve"
                    fig_roc, ax_roc = plt.subplots(figsize=(3, 2.8))
                    ax_roc.plot(fpr, tpr, color="#355E8E", lw=2, label="ROC")
                    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
                    ax_roc.set_xlabel("False Positive Rate")
                    ax_roc.set_ylabel("True Positive Rate")
                    ax_roc.set_title(title_str)
                    ax_roc.legend(fontsize=8)
                    ax_roc.set_xlim([0.0, 1.0])
                    ax_roc.set_ylim([0.0, 1.05])
                    plt.tight_layout(pad=0.3)
                    st.pyplot(fig_roc, use_container_width=True)
                    plt.close(fig_roc)
                except Exception:
                    st.caption("ROC curve unavailable.")
            else:
                st.caption("No ROC curve data.")

        # 3. Precision-Recall Curve
        with dc3:
            if pr_data is not None:
                try:
                    prec_vals = [v for v in pr_data.get("precision", []) if v is not None]
                    rec_vals  = [v for v in pr_data.get("recall",    []) if v is not None]
                    ap_val    = final.get("ap") if final else None
                    title_str = f"Precision-Recall  (AP={ap_val:.4f})" if ap_val is not None else "Precision-Recall Curve"
                    fig_pr, ax_pr = plt.subplots(figsize=(3, 2.8))
                    ax_pr.plot(rec_vals, prec_vals, color="#4A7BA5", lw=2)
                    ax_pr.set_xlabel("Recall")
                    ax_pr.set_ylabel("Precision")
                    ax_pr.set_title(title_str)
                    ax_pr.set_xlim([0.0, 1.0])
                    ax_pr.set_ylim([0.0, 1.05])
                    plt.tight_layout(pad=0.3)
                    st.pyplot(fig_pr, use_container_width=True)
                    plt.close(fig_pr)
                except Exception:
                    st.caption("PR curve unavailable.")
            else:
                st.caption("No PR curve data.")

        # 4. Probability Distribution
        with dc4:
            if hist_data is not None:
                try:
                    counts = hist_data.get("counts", [])
                    bins   = [v for v in hist_data.get("bins", []) if v is not None]
                    if counts and len(bins) == len(counts) + 1:
                        bin_centers = [(bins[j] + bins[j + 1]) / 2 for j in range(len(counts))]
                        colors = ["#4A7BA5" if c < 0.5 else "#E87040" for c in bin_centers]
                        fig_ph, ax_ph = plt.subplots(figsize=(3, 2.8))
                        ax_ph.bar(bin_centers, counts, width=(bins[1] - bins[0]) * 0.9,
                                  color=colors, edgecolor="white", linewidth=0.5)
                        ax_ph.axvline(0.5, color="gray", linestyle="--", lw=1.2, alpha=0.7)
                        ax_ph.set_xlabel("Predicted Probability")
                        ax_ph.set_ylabel("Count")
                        ax_ph.set_title("Probability Distribution")
                        # Legend
                        import matplotlib.patches as mpatches
                        p0 = mpatches.Patch(color="#4A7BA5", label="Pred 0 (<0.5)")
                        p1 = mpatches.Patch(color="#E87040", label="Pred 1 (≥0.5)")
                        ax_ph.legend(handles=[p0, p1], fontsize=8)
                        plt.tight_layout(pad=0.3)
                        st.pyplot(fig_ph, use_container_width=True)
                        plt.close(fig_ph)
                    else:
                        st.caption("Probability histogram data malformed.")
                except Exception:
                    st.caption("Probability histogram unavailable.")
            else:
                st.caption("No probability histogram data.")

    # ── Dataset Overview ──────────────────────────────────────────────────────
    st.divider()
    with st.expander("Dataset Overview"):
        try:
            ds_r = requests.get(_endpoint("dataset_stats"), timeout=15)
            if ds_r.ok:
                ds      = ds_r.json()
                n_total = ds.get("n_total", 0)
                n_pos   = ds.get("n_positive")
                n_neg   = ds.get("n_negative")

                c1, c2, c3 = st.columns(3)
                c1.metric("Total samples", f"{n_total:,}")
                if n_pos is not None:
                    c2.metric("Positive (label=1)", f"{n_pos:,}")
                    c3.metric("Negative (label=0)", f"{n_neg:,}")

                hists = ds.get("length_hists", {})
                if hists:
                    import plotly.graph_objects as go
                    h_cols = st.columns(len(hists))
                    for idx, (lbl, hist) in enumerate(hists.items()):
                        bins   = hist.get("bins", [])
                        counts = hist.get("counts", [])
                        if bins and counts:
                            centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(counts))]
                            fig_h = go.Figure(go.Bar(
                                x=centers, y=counts, marker_color="#4A7BA5",
                            ))
                            fig_h.update_layout(
                                title=f"{lbl} length",
                                xaxis_title="Length (chars)",
                                yaxis_title="Count",
                                height=260,
                                margin=dict(l=20, r=20, t=36, b=30),
                                template=plotly_template,
                            )
                            h_cols[idx].plotly_chart(fig_h, use_container_width=True)
            else:
                st.caption("Dataset stats not available.")
        except Exception:
            st.caption("Could not load dataset stats.")

    # ── Embedding & Model Feature Space Visualization (UMAP) ─────────────────────
    st.divider()
    st.subheader("Embedding Space Visualization")
    st.caption(
        "Left: raw pair embedding space colored by label. "
        "Right: model penultimate-layer space colored by TP / TN / FP / FN. "
        "PCA pre-reduction applied automatically for ≥ 100 samples."
    )

    key_prefix = "example" if active_is_example else "run"
    umap_key       = f"{key_prefix}_umap_{rid}"
    model_umap_key = f"{key_prefix}_model_umap_{rid}"

    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        umap_label = "Load Embedding UMAP" if active_is_example else "Generate Embedding UMAP"
        if st.button(umap_label, key="btn_umap"):
            spinner_text = "Loading cached embedding UMAP..." if active_is_example else "Computing embedding UMAP..."
            with st.spinner(spinner_text):
                try:
                    r = requests.get(_endpoint("umap_data"), timeout=300)
                    if r.ok:
                        st.session_state[umap_key] = r.json()
                    else:
                        try:
                            d = r.json()
                            err = d.get("error") or d.get("detail") or str(d)
                        except Exception:
                            err = r.text or "Unknown error"
                        st.error(f"Backend error ({r.status_code}): {err}")
                except Exception as exc:
                    st.error(f"Request error: {exc}")

    with btn_c2:
        model_umap_label = "Load Model Feature UMAP" if active_is_example else "Generate Model Feature UMAP"
        if st.button(model_umap_label, key="btn_model_umap"):
            spinner_text = (
                "Loading cached model feature UMAP..."
                if active_is_example
                else "Extracting penultimate-layer features & computing UMAP..."
            )
            with st.spinner(spinner_text):
                try:
                    r = requests.get(_endpoint("model_umap"), timeout=300)
                    if r.ok:
                        st.session_state[model_umap_key] = r.json()
                    else:
                        try:
                            d = r.json()
                            err = d.get("error") or d.get("detail") or str(d)
                        except Exception:
                            err = r.text or "Unknown error"
                        st.error(f"Backend error ({r.status_code}): {err}")
                except Exception as exc:
                    st.error(f"Request error: {exc}")

    import plotly.graph_objects as go

    def _render_embedding_umap(payload):
        xs   = payload["x"]
        ys   = payload["y"]
        lbls = payload["labels"]
        spls = payload.get("splits", ["train"] * len(xs))
        n_vis = payload["n_samples"]

        colour_map = {1: "#E87040", 0: "#355E8E", -1: "#aaaaaa"}
        symbol_map = {"train": "circle", "val": "diamond"}
        group_name = {
            ("train", 1): "Train · Positive",
            ("train", 0): "Train · Negative",
            ("val",   1): "Val · Positive",
            ("val",   0): "Val · Negative",
            ("train",-1): "Train · Unknown",
            ("val",  -1): "Val · Unknown",
        }
        traces: dict = {}
        for x, y, l, s in zip(xs, ys, lbls, spls):
            key = (s, l)
            traces.setdefault(key, {"x": [], "y": []})
            traces[key]["x"].append(x)
            traces[key]["y"].append(y)

        order = [("train", 0), ("train", 1), ("val", 0), ("val", 1),
                 ("train", -1), ("val", -1)]
        fig = go.Figure()
        for key in order:
            if key not in traces:
                continue
            spl, lbl = key
            fig.add_trace(go.Scatter(
                x=traces[key]["x"], y=traces[key]["y"], mode="markers",
                name=group_name.get(key, str(key)),
                marker=dict(
                    size=5 if spl == "val" else 4,
                    color=colour_map.get(lbl, "#888"),
                    symbol=symbol_map.get(spl, "circle"),
                    opacity=0.85 if spl == "val" else 0.5,
                    line=dict(width=0.5, color="white") if spl == "val" else dict(width=0),
                ),
            ))
        fig.update_layout(
            title=f"Raw Embedding Space  (n={n_vis:,})  ●train  ◆val",
            xaxis_title="UMAP 1", yaxis_title="UMAP 2",
            height=480, template=plotly_template,
            legend=dict(title="Split · Label"),
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_model_umap(payload):
        xs, ys, cats = payload["x"], payload["y"], payload["categories"]
        n_vis = payload["n_samples"]
        colour_map = {"TP": "#2ecc71", "TN": "#3498db", "FP": "#e74c3c", "FN": "#f39c12", "Unknown": "#aaaaaa"}
        symbol_map = {"TP": "diamond", "TN": "diamond", "FP": "diamond", "FN": "diamond", "Unknown": "circle"}
        traces: dict = {}
        for x, y, c in zip(xs, ys, cats):
            traces.setdefault(c, {"x": [], "y": []})
            traces[c]["x"].append(x)
            traces[c]["y"].append(y)
        fig = go.Figure()
        for c in ["TP", "TN", "FP", "FN", "Unknown"]:
            if c not in traces:
                continue
            fig.add_trace(go.Scatter(
                x=traces[c]["x"], y=traces[c]["y"], mode="markers",
                name=c,
                marker=dict(
                    size=6 if c != "Unknown" else 4,
                    color=colour_map[c],
                    symbol=symbol_map.get(c, "diamond"),
                    opacity=0.8 if c != "Unknown" else 0.6,
                ),
            ))
        fig.update_layout(
            title=f"Model Feature Space  (n={n_vis:,})",
            xaxis_title="UMAP 1", yaxis_title="UMAP 2",
            height=480, template=plotly_template, legend=dict(title="Category"),
        )
        st.plotly_chart(fig, use_container_width=True)

    emb_payload   = st.session_state.get(umap_key)
    model_payload = st.session_state.get(model_umap_key)

    if emb_payload and model_payload:
        col_l, col_r = st.columns(2)
        with col_l:
            _render_embedding_umap(emb_payload)
        with col_r:
            _render_model_umap(model_payload)
    elif emb_payload:
        _render_embedding_umap(emb_payload)
    elif model_payload:
        _render_model_umap(model_payload)

    if not active_is_example:
        st.divider()
        d1, d2, d3 = st.columns(3)
        with d1:
            resp = requests.get(f"{BACKEND}/download_embedding/{rid}", stream=True)
            if resp.status_code == 200:
                st.download_button("Download embeddings (.pkl)", data=resp.content,
                                   file_name=f"embedding_{rid}.pkl",
                                   mime="application/octet-stream")
        with d2:
            resp = requests.get(f"{BACKEND}/download_model/{rid}", stream=True)
            if resp.status_code == 200:
                st.download_button("Download model weights (.pt)", data=resp.content,
                                   file_name=f"model_{rid}.pt",
                                   mime="application/octet-stream")
        with d3:
            resp = requests.get(f"{BACKEND}/download_bundle/{rid}", stream=True)
            if resp.status_code == 200:
                st.download_button("Download artifact bundle (.zip)", data=resp.content,
                                   file_name=f"artifacts_{rid}.zip",
                                   mime="application/zip")

elif status == "cancelled":
    st.warning("This job was cancelled.")

elif status == "failed":
    st.error(f"Job failed: {status_data.get('result', 'check backend logs')}")

# ── cancel form ───────────────────────────────────────────────────────────────
if status in ("running", "queued") and not active_is_example:
    st.divider()
    with st.expander("Cancel this job"):
        st.caption(
            "Enter the cancel token shown at submission time. "
            "Wrong token → request is rejected, job continues."
        )
        prefill = default_token if rid == default_id else ""
        token_input = st.text_input("Cancel token", value=prefill,
                                    type="password", key="cr_cancel_token")
        if st.button("Cancel job", type="secondary"):
            if not token_input.strip():
                st.error("Enter the cancel token.")
            else:
                try:
                    cr = requests.post(
                        f"{BACKEND}/cancel_job/{rid}",
                        json={"cancel_token": token_input.strip()},
                        timeout=10,
                    )
                    result = cr.json()
                    if cr.status_code == 200:
                        st.success("Job cancelled.")
                        st.session_state.pop("last_cancel_token", None)
                        st.rerun()
                    else:
                        st.error(result.get("error", "Cancel failed."))
                except Exception as e:
                    st.error(f"Could not reach backend: {e}")

    if auto_refresh:
        time.sleep(4)
        st.rerun()
