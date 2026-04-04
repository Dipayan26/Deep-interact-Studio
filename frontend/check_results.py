import os
import time

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

col_check, col_auto = st.columns([1, 2])
with col_check:
    check_btn = st.button("Check Status", type="primary")
with col_auto:
    auto_refresh = st.checkbox("Auto-refresh while training", value=False)

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
status_colours = {
    "completed": "green",
    "running":   "blue",
    "queued":    "orange",
    "failed":    "red",
    "cancelled": "gray",
}
st.markdown(f"**Status:** :{status_colours.get(status, 'gray')}[{status.capitalize()}]")

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
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Val Accuracy", f"{final['val_acc']:.4f}"   if final.get("val_acc")   is not None else "—")
        m2.metric("AUROC",        f"{final['auroc']:.4f}"     if final.get("auroc")     is not None else "—")
        m3.metric("Precision",    f"{final['precision']:.4f}" if final.get("precision") is not None else "—")
        m4.metric("Recall",       f"{final['recall']:.4f}"    if final.get("recall")    is not None else "—")
        m5.metric("F1",           f"{final['f1']:.4f}"        if final.get("f1")        is not None else "—")

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
