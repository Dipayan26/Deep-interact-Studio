import streamlit as st
import pandas as pd
import requests
import numpy as np
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8005")


st.title('Build Page')


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

st.button("RUN")
