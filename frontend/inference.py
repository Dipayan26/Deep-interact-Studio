import io
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

st.title("Inference-tab")
st.caption("Run a trained DL model on new protein pairs.")

st.divider()

# ── Select trained model ───────────────────────────────────────────────────────
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

job_options = {
    f"{j['run_id']}  (acc={j.get('val_acc') or '—'}  auroc={j.get('auroc') or '—'})": j["run_id"]
    for j in completed_training
}
selected_label = st.selectbox("Training run", list(job_options.keys()), label_visibility="collapsed")
source_run_id  = job_options[selected_label]
st.caption(f"Source run ID: `{source_run_id}`")

st.divider()

# ── Upload inference CSV ───────────────────────────────────────────────────────
st.subheader("Input Data")
st.markdown(
    "Upload a CSV with columns **`proteinA`** and **`proteinB`**. No label column required."
)
infer_file = st.file_uploader("Select CSV file", type=["csv"], label_visibility="collapsed")

st.divider()

if st.button("Run Inference", type="primary"):
    if infer_file is None:
        st.error("No file selected.")
    else:
        try:
            infer_df = pd.read_csv(io.BytesIO(infer_file.getvalue()))
            missing  = [c for c in ("proteinA", "proteinB") if c not in infer_df.columns]
            if missing:
                st.error(
                    f"CSV is missing required column(s): {', '.join(f'`{c}`' for c in missing)}. "
                    "Rename your columns to `proteinA` and `proteinB`."
                )
                st.stop()
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()

        with st.spinner("Submitting inference job..."):
            try:
                files = [("files", (infer_file.name, infer_file.getvalue(), "text/csv"))]
                r = requests.post(f"{BACKEND}/run_inference/{source_run_id}", files=files)
                r.raise_for_status()
                data = r.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    infer_run_id = data["run_id"]
                    st.success(f"Inference job submitted — Run ID: `{infer_run_id}`")
                    st.session_state["infer_run_id"] = infer_run_id
            except Exception as e:
                st.error(f"Submission failed: {e}")

# ── Poll, download, and visualise ──────────────────────────────────────────────
infer_run_id = st.session_state.get("infer_run_id", "")
if infer_run_id:
    st.divider()
    st.subheader("Results")

    rid_input = st.text_input("Inference Run ID", value=infer_run_id)
    rid       = rid_input.strip()

    col_check, col_auto = st.columns([1, 2])
    with col_check:
        check_btn = st.button("Check Status", key="infer_check")
    with col_auto:
        auto = st.checkbox("Auto-refresh", value=False, key="infer_auto")

    if check_btn or auto:
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
        st.write(f"**Status:** {status.capitalize()}")

        if status == "completed":
            resp = requests.get(f"{BACKEND}/download_results/{rid}", stream=True)
            if resp.status_code == 200:
                csv_bytes = resp.content
                st.download_button(
                    "Download results (.csv)",
                    data=csv_bytes,
                    file_name=f"ppi_results_{rid}.csv",
                    mime="text/csv",
                )

                # Parse results and show visualisations
                try:
                    results_df = pd.read_csv(io.BytesIO(csv_bytes))
                    # Filter rows that have a valid probability
                    valid_df = results_df.dropna(subset=["probability"])
                    valid_df = valid_df[pd.to_numeric(valid_df["probability"], errors="coerce").notna()].copy()
                    valid_df["probability"] = valid_df["probability"].astype(float)
                    valid_df["prediction"]  = valid_df["prediction"].astype(int) if "prediction" in valid_df.columns else (valid_df["probability"] >= 0.5).astype(int)

                    if len(valid_df) > 0:
                        st.divider()
                        st.subheader("Result Visualisations")

                        vis_c1, vis_c2, vis_c3 = st.columns(3)

                        # 1. Probability histogram
                        with vis_c1:
                            try:
                                probs = valid_df["probability"].values
                                counts, bin_edges = np.histogram(probs, bins=20, range=(0.0, 1.0))
                                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                                colors = ["#4A7BA5" if c < 0.5 else "#E87040" for c in bin_centers]

                                fig_h, ax_h = plt.subplots(figsize=(4, 3.5))
                                ax_h.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]) * 0.9,
                                         color=colors, edgecolor="white", linewidth=0.5)
                                ax_h.axvline(0.5, color="gray", linestyle="--", lw=1.2, alpha=0.7)
                                ax_h.set_xlabel("Predicted Probability")
                                ax_h.set_ylabel("Count")
                                ax_h.set_title("Probability Distribution")
                                import matplotlib.patches as mpatches
                                p0 = mpatches.Patch(color="#4A7BA5", label="Pred 0")
                                p1 = mpatches.Patch(color="#E87040", label="Pred 1")
                                ax_h.legend(handles=[p0, p1], fontsize=8)
                                plt.tight_layout(pad=0.3)
                                st.pyplot(fig_h, use_container_width=True)
                                plt.close(fig_h)
                            except Exception:
                                st.caption("Histogram unavailable.")

                        # 2. Prediction summary (donut chart)
                        with vis_c2:
                            try:
                                n_pos = int((valid_df["prediction"] == 1).sum())
                                n_neg = int((valid_df["prediction"] == 0).sum())
                                total = n_pos + n_neg

                                fig_d, ax_d = plt.subplots(figsize=(4, 3.5))
                                wedge_sizes  = [n_neg, n_pos]
                                wedge_labels = [f"Pred 0\n({n_neg})", f"Pred 1\n({n_pos})"]
                                wedge_colors = ["#4A7BA5", "#E87040"]
                                wedges, texts, autotexts = ax_d.pie(
                                    wedge_sizes,
                                    labels=wedge_labels,
                                    colors=wedge_colors,
                                    autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
                                    startangle=90,
                                    wedgeprops={"width": 0.5},
                                    textprops={"fontsize": 9},
                                )
                                for at in autotexts:
                                    at.set_fontsize(8)
                                ax_d.set_title(f"Prediction Summary\n(n={total})")
                                plt.tight_layout(pad=0.3)
                                st.pyplot(fig_d, use_container_width=True)
                                plt.close(fig_d)
                            except Exception:
                                st.caption("Prediction summary unavailable.")

                        # 3. Confidence scatter (sorted bar chart)
                        with vis_c3:
                            try:
                                sorted_df = valid_df.sort_values("probability").reset_index(drop=True)
                                n_show = min(len(sorted_df), 200)
                                sorted_df = sorted_df.iloc[:n_show] if len(sorted_df) > n_show else sorted_df
                                bar_colors = ["#4A7BA5" if p < 0.5 else "#E87040"
                                              for p in sorted_df["probability"].values]

                                fig_s, ax_s = plt.subplots(figsize=(4, 3.5))
                                ax_s.bar(range(len(sorted_df)), sorted_df["probability"].values,
                                         color=bar_colors, width=1.0, edgecolor="none")
                                ax_s.axhline(0.5, color="gray", linestyle="--", lw=1.2, alpha=0.7)
                                ax_s.set_xlabel("Pair (sorted by probability)")
                                ax_s.set_ylabel("Probability")
                                ax_s.set_title(
                                    f"Confidence Scores"
                                    + (f" (top {n_show})" if n_show < len(valid_df) else "")
                                )
                                ax_s.set_ylim(0, 1.05)
                                import matplotlib.patches as mpatches
                                p0 = mpatches.Patch(color="#4A7BA5", label="Pred 0")
                                p1 = mpatches.Patch(color="#E87040", label="Pred 1")
                                ax_s.legend(handles=[p0, p1], fontsize=8)
                                plt.tight_layout(pad=0.3)
                                st.pyplot(fig_s, use_container_width=True)
                                plt.close(fig_s)
                            except Exception:
                                st.caption("Confidence scatter unavailable.")

                except Exception:
                    pass  # Visualisations are optional — don't block download button

            else:
                st.warning("Results file not available yet.")

        elif status == "failed":
            st.error(f"Inference failed: {sd.get('result', 'check backend logs')}")

        elif status in ("running", "queued") and auto:
            time.sleep(4)
            st.rerun()
