
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
if menu == "Submit Training":
    st.subheader("Submit Training Job")

    seq = st.text_area("Enter Protein Sequence (A,C,G,T or AA sequence):")

    if st.button("Train Model"):
        if seq.strip() == "":
            st.error("Enter a valid sequence")
        else:
            r = requests.post(f"{BACKEND}/create_job", json={"sequence": seq})
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
                # st.write("Prediction:", data["result"]["prediction"])
                st.write("Prediction:", data["result"])
                # st.write("Probability:", data["result"]["probability"])


























