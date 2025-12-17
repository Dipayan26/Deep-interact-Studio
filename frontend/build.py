
## Frontend file Build.py
###################################
import streamlit as st
import pandas as pd
import requests
import numpy as np
import os

##"BACKEND URL" is the   - BACKEND_URL=http://backend:8005 present in docker compose file 
# BACKEND_URL = os.getenv("BACKEND_URL")

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8005")


st.title('Build Page')
######################################

BACKEND = BACKEND_URL

st.title("PPI Prediction - Training Demo (Real Deep Learning)")

menu = st.sidebar.selectbox("Menu", ["Submit Training", "Check Results"])

# --------------------------------------------------------
# SUBMIT TRAINING JOB
# --------------------------------------------------------
# if menu == "Submit Training":
#     st.subheader("Submit Training Job")

#     seq = st.text_area("Enter Protein Sequence (A,C,G,T or AA sequence):")

#     if st.button("Train Model"):
#         if seq.strip() == "":
#             st.error("Enter a valid sequence")
#         else:
#             r = requests.post(f"{BACKEND}/create_job", json={"sequence": seq})
#             data = r.json()
#             st.success(f"Run ID: {data['run_id']}")
#             st.info("Save this Run ID and check results later.")



if menu == "Submit Training":
    st.subheader("Submit Training Job")

    seq_file = st.file_uploader(
        "Upload Protein Sequence File",
        type=["csv"]
    )

    if st.button("Train Model"):
        if seq_file is None:
            st.error("Upload a valid CSV file")
        else:
            files = [
                (
                    "files",  # must match backend argument name
                    (seq_file.name, seq_file.getvalue(), "text/csv")
                )
            ]

            r = requests.post(
                f"{BACKEND}/create_job",
                files=files
            )
            data = r.json()
            st.success(f"Run ID: {data['run_id']}")
            st.info("Save this Run ID and check results later.")










# --------------------------------------------------------
# CHECK RESULTS
# --------------------------------------------------------
if menu == "Check Results":
    st.subheader("Check Training Results")

    run_id = st.text_input("Enter Run ID")

    if st.button("Check Status"):
        r = requests.get(f"{BACKEND}/check_status/{run_id}")
        data = r.json()

        if "error" in data:
            st.error("Invalid Run ID")
        else:
            st.write("Status:", data["status"])

        if data["status"] == "completed":

            st.success("Training Completed!")
            # st.write("Prediction:", data["result"])

            download_url = f"{BACKEND}/download_embedding/{run_id}"

            # Stream download safely (handles >50MB files)
            with requests.get(download_url, stream=True) as r:
                if r.status_code != 200:
                    st.error("Embedding file not found in backend.")
                else:
                    file_bytes = r.content   # read streamed content into memory

                    st.download_button(
                        label="Download Embedding File",
                        data=file_bytes,
                        file_name=f"embedding_{run_id}.pkl",
                        mime="application/octet-stream",
                    )






















