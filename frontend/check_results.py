import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://backend:8005")

st.title("Check Results")
st.caption("Monitor training progress and download artefacts.")

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
        st.rerun()

if not (check_btn or auto_refresh):
    st.stop()

rid = (run_id or "").strip()
if not rid:
    st.error("Enter a run ID.")
    st.stop()

try:
    status_r    = requests.get(f"{BACKEND}/check_status/{rid}", timeout=5)
    status_data = status_r.json()
except Exception as e:
    st.error(f"Could not reach backend: {e}")
    st.stop()

try:
    metrics_r    = requests.get(f"{BACKEND}/metrics/{rid}", timeout=5)
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

# ── Model details expander ─────────────────────────────────────────────────────
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
    total += cur + 1  # output layer
    return total

if hp:
    with st.expander("Model details"):
        if task_type == "dti":
            chem_dim = int(hp.get("chem_dim", 768))
            esm_dim  = int(hp.get("esm_dim",  480))
            input_dim = chem_dim + esm_dim

            chem_model = hp.get("chem_model", "seyonec/ChemBERTa-zinc-base-v1")
            esm_model  = hp.get("esm_model",  "esm2_t12_35M_UR50D")
            chem_label = chem_model.split("/")[-1] if "/" in chem_model else chem_model
            esm_label  = esm_model

            layer_configs = hp.get("layer_configs", [])
            n_params = _approx_params(input_dim, layer_configs) if layer_configs else None

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Embedding model**")
                st.caption(f"ChemBERTa `{chem_label}` {chem_dim}-dim + ESM2 `{esm_label}` {esm_dim}-dim")
                st.markdown("**Input dim**")
                st.caption(f"{input_dim:,} ({chem_dim} chem + {esm_dim} prot)")
            with col_b:
                if n_params is not None:
                    st.markdown("**Approx. parameters**")
                    st.caption(f"{n_params:,}")
                pair_mode = hp.get("pair_mode")
                if pair_mode:
                    st.markdown("**Pair mode**")
                    st.caption(pair_mode)
        else:
            esm_model = hp.get("esm_model", "esm2_t12_35M_UR50D")
            esm_dim   = int(hp.get("esm_dim", 480))
            pair_mode = hp.get("pair_mode", "all")
            input_dim = int(hp.get("input_dim", esm_dim))
            layer_configs = hp.get("layer_configs", [])
            n_params = _approx_params(input_dim, layer_configs) if layer_configs else None

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Embedding model**")
                st.caption(f"ESM2 `{esm_model}` {esm_dim}-dim")
                st.markdown("**Pair representation**")
                st.caption(pair_mode)
            with col_b:
                st.markdown("**Input dim**")
                st.caption(f"{input_dim:,}")
                if n_params is not None:
                    st.markdown("**Approx. parameters**")
                    st.caption(f"{n_params:,}")

# ── progress bar ──────────────────────────────────────────────────────────────
epoch        = metrics_data.get("epoch", 0)
total_epochs = metrics_data.get("total_epochs", 0)
if total_epochs > 0:
    label = f"Epoch {epoch} / {total_epochs}"
    if metrics_data.get("early_stopped"):
        label += "  (early stopped)"
    st.progress(epoch / total_epochs, text=label)

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

    has_diag = any(x is not None for x in [cm_data, roc_data, pr_data, hist_data])
    if has_diag:
        st.divider()
        st.subheader("Diagnostics")

        row1_c1, row1_c2 = st.columns(2)
        row2_c1, row2_c2 = st.columns(2)

        # 1. Confusion Matrix
        with row1_c1:
            if cm_data is not None:
                try:
                    cm_arr = np.array(cm_data, dtype=int)   # [[TN,FP],[FN,TP]]
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
                    im = ax_cm.imshow(cm_arr, cmap="Blues", aspect="auto")
                    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
                    ax_cm.set_xticks([0, 1])
                    ax_cm.set_yticks([0, 1])
                    ax_cm.set_xticklabels(["Pred 0", "Pred 1"])
                    ax_cm.set_yticklabels(["True 0", "True 1"])
                    ax_cm.set_title("Confusion Matrix")
                    # Annotate cells
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
        with row1_c2:
            if roc_data is not None:
                try:
                    fpr = [v for v in roc_data.get("fpr", []) if v is not None]
                    tpr = [v for v in roc_data.get("tpr", []) if v is not None]
                    auroc_val = final.get("auroc") if final else None
                    title_str = f"ROC Curve  (AUROC={auroc_val:.4f})" if auroc_val is not None else "ROC Curve"
                    fig_roc, ax_roc = plt.subplots(figsize=(4, 3.5))
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
        with row2_c1:
            if pr_data is not None:
                try:
                    prec_vals = [v for v in pr_data.get("precision", []) if v is not None]
                    rec_vals  = [v for v in pr_data.get("recall",    []) if v is not None]
                    ap_val    = final.get("ap") if final else None
                    title_str = f"Precision-Recall  (AP={ap_val:.4f})" if ap_val is not None else "Precision-Recall Curve"
                    fig_pr, ax_pr = plt.subplots(figsize=(4, 3.5))
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
        with row2_c2:
            if hist_data is not None:
                try:
                    counts = hist_data.get("counts", [])
                    bins   = [v for v in hist_data.get("bins", []) if v is not None]
                    if counts and len(bins) == len(counts) + 1:
                        bin_centers = [(bins[j] + bins[j + 1]) / 2 for j in range(len(counts))]
                        colors = ["#4A7BA5" if c < 0.5 else "#E87040" for c in bin_centers]
                        fig_ph, ax_ph = plt.subplots(figsize=(4, 3.5))
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

    st.divider()
    d1, d2 = st.columns(2)
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

elif status == "cancelled":
    st.warning("This job was cancelled.")

elif status == "failed":
    st.error(f"Job failed: {status_data.get('result', 'check backend logs')}")

# ── cancel form ───────────────────────────────────────────────────────────────
if status in ("running", "queued"):
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
