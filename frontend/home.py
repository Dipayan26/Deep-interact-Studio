import streamlit as st
import os
import requests

st.title('Home Page')
# home page documents 

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
# st.set_page_config(
#     page_title="Sequence-based PPI Neural Network Trainer",
#     layout="wide"
# )

# -------------------------------------------------
# Title Section
# -------------------------------------------------
st.title("🧬 Sequence-based Protein–Protein Interaction Prediction")
st.subheader("Neural Network Training Platform")

st.markdown(
    """
    Protein–Protein Interactions (PPIs) are central to almost all cellular processes.
    This platform is designed to **train deep learning models for PPI prediction using
    only protein sequence information**, without relying on structural or functional annotations.
    """
)

st.divider()

# -------------------------------------------------
# Motivation
# -------------------------------------------------
st.header("📌 Motivation")

st.markdown(
    """
    Experimental identification of PPIs is expensive, time-consuming, and incomplete.
    Sequence-based computational approaches provide a scalable alternative by learning
    interaction patterns directly from amino acid sequences using neural networks.

    Recent advances in deep learning — particularly **transformers and large protein language models** —
    have significantly improved sequence representation learning, making high-accuracy
    PPI prediction possible from sequence data alone.
    """
)

st.divider()

# -------------------------------------------------
# What This Platform Does
# -------------------------------------------------
st.header("⚙️ What This Platform Provides")

st.markdown(
    """
    This platform focuses **exclusively on model training**, and enables:

    - Training neural networks using **raw protein sequences**
    - Support for **custom and pretrained sequence encoders**
    - Flexible **protein–protein interaction modeling strategies**
    - Research-oriented experimentation with architectures and loss functions
    - Reproducible deep learning workflows for PPI prediction
    """
)

st.divider()

# -------------------------------------------------
# Model Architecture Overview
# -------------------------------------------------
st.header("🧠 Model Architecture Overview")

st.markdown(
    """
    A typical sequence-based PPI model trained using this platform follows three stages:

    **1. Protein Encoding**  
    Each protein sequence is converted into a numerical representation using neural
    encoders such as CNNs, recurrent networks, transformers, or pretrained protein
    language models (e.g., ESM).

    **2. Interaction Modeling**  
    Encoded protein representations are combined to capture interaction-specific features
    using mathematical operations or attention-based mechanisms.

    **3. Prediction Head**  
    A neural classifier predicts the likelihood or strength of interaction between
    protein pairs.
    """
)

st.divider()

# -------------------------------------------------
# Research Focus
# -------------------------------------------------
st.header("🔬 Research Focus")

st.markdown(
    """
    This tool is intended for **research and methodological development**, including:

    - Architecture design for sequence-based PPI prediction
    - Analysis of representation learning from protein sequences
    - Handling data imbalance in PPI datasets
    - Benchmarking different neural network components
    - Developing publishable deep learning methods for interactomics
    """
)

st.divider()

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.caption(
    "Sequence-based PPI Neural Network Training Platform | Research Use Only"
)
