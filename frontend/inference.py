import os
import time
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://backend:8005")

st.title("Inference")
st.caption("Run a trained model on new protein pairs.")

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
selected_label  = st.selectbox("Training run", list(job_options.keys()), label_visibility="collapsed")
source_run_id   = job_options[selected_label]
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

# ── Poll and download ──────────────────────────────────────────────────────────
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
                st.download_button(
                    "Download results (.csv)",
                    data=resp.content,
                    file_name=f"ppi_results_{rid}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("Results file not available yet.")

        elif status == "failed":
            st.error(f"Inference failed: {sd.get('result', 'check backend logs')}")

        elif status in ("running", "queued") and auto:
            time.sleep(4)
            st.rerun()
