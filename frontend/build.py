import streamlit as st
import pandas as pd
import requests
import numpy as np
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8005")


st.title('Build Page')


uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if st.button("Send to Backend"):
    if uploaded_file is None:
        st.error("Please upload a CSV file first.")
    else:
        # Send file to backend
        response = requests.post(
            f"{BACKEND_URL}/process_csv",
            files={"file": uploaded_file}
        )

        if response.status_code == 200:
            result = response.json()
            st.success("Processing completed!")

            st.write("### Result from Backend")
            st.json(result)
        else:
            st.error("Backend error")