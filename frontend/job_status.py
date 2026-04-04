import os
import requests
import pandas as pd
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://backend:8005")

st.title("Job Status")
st.caption("All submitted jobs — training and inference.")

st.divider()

col_refresh, _ = st.columns([1, 6])
with col_refresh:
    if st.button("Refresh"):
        st.rerun()

try:
    r = requests.get(f"{BACKEND}/jobs", timeout=5)
    r.raise_for_status()
    jobs = r.json()

    if not jobs:
        st.info("No jobs submitted yet.")
        st.stop()

    df = pd.DataFrame(jobs)
    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False)

    # display table
    display_cols = {
        "run_id":     "Run ID",
        "job_type":   "Type",
        "status":     "Status",
        "created_at": "Submitted",
        "val_acc":    "Val Acc",
        "auroc":      "AUROC",
        "f1":         "F1",
    }
    show_df = df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols)
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    # ── Detail panel ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Job Details")

    selected = st.selectbox("Select run ID", df["run_id"].tolist(), label_visibility="collapsed")

    if st.button("View details"):
        row = df[df["run_id"] == selected].iloc[0]
        st.write(f"**Run ID:** `{row['run_id']}`")
        st.write(f"**Type:** {row.get('job_type', '—')}")
        st.write(f"**Status:** {row['status']}")
        st.write(f"**Submitted:** {row.get('created_at', '—')}")
        if row.get("source_run_id"):
            st.write(f"**Source training run:** `{row['source_run_id']}`")

        # metrics chart for training jobs
        if row.get("job_type") == "train" and row["status"] in ("running", "completed"):
            try:
                mr = requests.get(f"{BACKEND}/metrics/{selected}", timeout=5)
                metrics = mr.json()
                history = metrics.get("history", {})
                if history.get("epoch"):
                    chart_df = pd.DataFrame({
                        "train_loss": history.get("train_loss", []),
                        "val_loss":   history.get("val_loss",   []),
                    }, index=history["epoch"])
                    st.line_chart(chart_df, x_label="Epoch", y_label="Loss")

                    final = metrics.get("final", {})
                    if final:
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Val Acc",   f"{final.get('val_acc', 0):.4f}"   if final.get('val_acc')   else "—")
                        c2.metric("AUROC",     f"{final.get('auroc', 0):.4f}"     if final.get('auroc')     else "—")
                        c3.metric("Precision", f"{final.get('precision', 0):.4f}" if final.get('precision') else "—")
                        c4.metric("Recall",    f"{final.get('recall', 0):.4f}"    if final.get('recall')    else "—")
                        c5.metric("F1",        f"{final.get('f1', 0):.4f}"        if final.get('f1')        else "—")
            except Exception:
                pass

        # downloads
        if row["status"] == "cancelled":
            st.warning("This job was cancelled by the user.")

        elif row["status"] == "completed":
            st.divider()
            col_a, col_b, col_c = st.columns(3)

            if row.get("job_type") == "train":
                with col_a:
                    resp = requests.get(f"{BACKEND}/download_embedding/{selected}", stream=True)
                    if resp.status_code == 200:
                        st.download_button("Embeddings (.pkl)", data=resp.content,
                                           file_name=f"embedding_{selected}.pkl",
                                           mime="application/octet-stream")
                with col_b:
                    resp = requests.get(f"{BACKEND}/download_model/{selected}", stream=True)
                    if resp.status_code == 200:
                        st.download_button("Model weights (.pt)", data=resp.content,
                                           file_name=f"model_{selected}.pt",
                                           mime="application/octet-stream")

            elif row.get("job_type") == "inference":
                with col_a:
                    resp = requests.get(f"{BACKEND}/download_results/{selected}", stream=True)
                    if resp.status_code == 200:
                        st.download_button("Results (.csv)", data=resp.content,
                                           file_name=f"results_{selected}.csv",
                                           mime="text/csv")

except Exception as e:
    st.error(f"Could not reach backend: {e}")
