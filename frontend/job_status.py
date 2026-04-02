import streamlit as st
import requests
import pandas as pd
import time
import os

BACKEND = os.getenv("BACKEND_URL", "http://backend:8005")

st.subheader("All Submitted Jobs")

# -------- Auto-refresh logic --------
# refresh_secs = 10

# # Initialize timestamp
# if "last_refresh" not in st.session_state:
#     st.session_state.last_refresh = time.time()

# # If X seconds passed → refresh page
# if time.time() - st.session_state.last_refresh >= refresh_secs:
#     st.session_state.last_refresh = time.time()
#     st.rerun()

# st.caption(f"Auto-refreshing every {refresh_secs} seconds...")

## refress button 
if st.button("🔄 Refresh Now"):
    st.session_state.last_refresh = 0   # force immediate refresh
    st.rerun()



# -------- Fetch Jobs from Backend --------
try:
    r = requests.get(f"{BACKEND}/jobs")
    jobs = r.json()

    df = pd.DataFrame(jobs)

    # Color coding
    def color_status(status):
        if status == "completed": return "🟢 completed"
        if status == "running": return "🔵 running"
        if status == "queued": return "🟡 queued"
        if status == "failed": return "🔴 failed"
        return status

    df["status"] = df["status"].apply(color_status)

    # Sort by newest
    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False)

    st.dataframe(df[["run_id", "status", "created_at"]], use_container_width=True)

    run_id = st.selectbox("Select job to view details:", df["run_id"].tolist())

    if st.button("View Job Details"):
        st.write(df[df["run_id"] == run_id].iloc[0])

        # Embedding download
        download_url = f"{BACKEND}/download_embedding/{run_id}"
        resp = requests.get(download_url, stream=True)

        if resp.status_code == 200:
            st.download_button(
                label="Download Embedding File",
                data=resp.content,
                file_name=f"embedding_{run_id}.pkl",
                mime="application/octet-stream",
            )
        else:
            st.info("Embedding file not generated yet.")
        
except Exception as e:
    st.error(f"Error fetching jobs: {e}")
