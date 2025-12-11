import os
import requests
import streamlit as st

# Backend URL inside Docker network
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8005")

st.title("Web3 Demo: Streamlit + FastAPI + Docker + Nginx")

st.write("This demo calls a FastAPI backend from Streamlit.")

col1, col2 = st.columns(2)

with col1:
    # a = st.number_input("a", value=1.0)
    a = st.text_input("a", value=1.0)

with col2:
    b = st.number_input("b", value=2.0)

if st.button("Compute a + b"):
    try:
        resp = requests.post(
            f"{BACKEND_URL}/sum",
            json={"a": a, "b": b},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        st.success(f"Result: {data['result']}")
    except Exception as e:
        st.error(f"Error contacting backend: {e}")

st.write("---")
# st.caption(f"Backend URL: {BACKEND_URL}")
