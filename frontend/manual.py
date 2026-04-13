import streamlit as st

st.title("Manual")
st.caption("Step-by-step guide for using Deep-Prot Studio.")
st.divider()

with st.expander("1 — Prepare your data", expanded=True):
    st.markdown("""
Upload a CSV file with your protein sequences and interaction labels.

| Column | Description |
|--------|-------------|
| `proteinA` | Amino acid sequence of the first protein |
| `proteinB` | Amino acid sequence of the second protein |
| `label` | Interaction label: `1` (interacting) or `0` (non-interacting) |

- Accepted characters: standard IUPAC amino acid letters.
- Sequences longer than 1,022 residues are handled via a sliding-window approach automatically.
- Maximum **3,000 pairs** per training job.
""")

with st.expander("2 — Configure embedding model"):
    st.markdown("""
Select an ESM2 model for generating protein sequence embeddings.

| Model | Embedding dim | Speed |
|-------|--------------|-------|
| ESM2 8M | 320 | Fastest |
| ESM2 35M | 480 | Default |
| ESM2 150M | 640 | Accurate |
| ESM2 650M | 1280 | Slow |

Larger models produce richer embeddings but require more GPU memory and time.
""")

with st.expander("3 — Build your model architecture"):
    st.markdown("""
Use the **Model Builder** to stack layers on top of the ESM2 embeddings.

Available layer types:

- **Linear** — fully connected layer with activation and optional batch normalisation.
- **CNN1D** — 1-D convolutional layer for local feature extraction.
- **BiLSTM** — bidirectional LSTM for sequence context.
- **GRU** — gated recurrent unit (bidirectional option available).
- **Transformer** — self-attention encoder block.
- **Residual** — skip-connection block that preserves input dimension.

Re-order layers with the ↑ / ↓ buttons. Remove layers with ✕.
The architecture preview updates live showing dimensions and approximate parameter count.
""")

with st.expander("4 — Set training parameters"):
    st.markdown("""
| Parameter | Description |
|-----------|-------------|
| **Epochs** | Number of full passes over the training set (5–100). |
| **Learning rate** | Adam optimiser step size (0.001 / 0.0005 / 0.0001). |
| **Batch size** | Samples per gradient update (32 / 64 / 128). |
| **Early stopping** | Stop if validation loss does not improve for N epochs (0 = disabled). |
| **Pairs to use** | Soft cap — subsample from uploaded data (max 3,000). |
| **Training split** | Fraction of selected pairs used for training; remainder for testing. |
""")

with st.expander("5 — Submit and monitor"):
    st.markdown("""
1. Click **Submit Training Job**.
2. Save the displayed **cancel token** — it will not be shown again.
3. Navigate to **Tools → Check Results** and enter your Run ID to monitor live progress (loss, accuracy, AUROC).
4. Once completed, download the trained model (`.pt`) and embeddings (`.pkl`).
""")

with st.expander("6 — Run inference"):
    st.markdown("""
Go to **Tools → Inference**.

- Enter the **Run ID** of a completed training job.
- **Single Pair** — paste two sequences (raw or FASTA) and click *Predict Interaction*.
- **Batch CSV** — upload a CSV with `proteinA` and `proteinB` columns.

Results include probability scores, prediction labels, and (when ground-truth labels are provided) ROC/PR curves, confusion matrix, and threshold sensitivity analysis.
""")
