import streamlit as st
import pandas as pd
import requests
import numpy as np
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8005")


st.title('Build Page')


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if st.button("Send CSV to Backend"):
    if uploaded_file is None:
        st.error("Please upload a CSV first.")
    else:
        try:
            # Send file directly (no need for name/getvalue/mimetype)
            files = {"file": uploaded_file}

            resp = requests.post(
                f"{BACKEND_URL}/process_csv",
                files=files,
                timeout=30,
            )
            resp.raise_for_status()

            st.success(resp.json())
        except Exception as e:
            st.error(e)
