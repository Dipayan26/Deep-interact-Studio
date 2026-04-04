import streamlit as st

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Deep-Prot Studio")
st.caption("GPU-accelerated deep learning platform for biological sequence prediction")

st.divider()

st.markdown("""
**Deep-Prot Studio** is an open-source platform for training and deploying deep learning models
on biological sequence data. It uses protein language model (PLM) embeddings as universal
sequence representations, enabling researchers to build, evaluate, and apply predictive models
across a range of molecular interaction and annotation tasks — without writing ML code.

Each task module shares the same infrastructure: asynchronous GPU training via Celery,
per-epoch live metrics, and downloadable artefacts (embeddings, weights, predictions).
The underlying encoder is **ESM2** (Meta AI), a transformer-based protein language model
trained on UniRef50.
""")

st.divider()

# ── Task Overview ─────────────────────────────────────────────────────────────
st.subheader("Model Building Tasks")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("**Protein–Protein Interaction**")
        st.caption("Predict whether two proteins physically interact, from sequence alone.")
        st.markdown("Input: protein pair CSV + labels")
        st.markdown("Output: interaction probability")
        st.success("Available")

    with st.container(border=True):
        st.markdown("**Drug–Target Interaction**")
        st.caption("Predict binding between a small molecule (SMILES) and a target protein.")
        st.markdown("Input: SMILES + protein sequence + labels")
        st.markdown("Output: binding probability")
        st.warning("Coming Soon")

with col2:
    with st.container(border=True):
        st.markdown("**Subcellular Localization**")
        st.caption("Classify where a protein resides within a cell across 10 compartments.")
        st.markdown("Input: protein sequence CSV + compartment labels")
        st.markdown("Output: localization class")
        st.warning("Coming Soon")

    with st.container(border=True):
        st.markdown("**RNA–Protein Interaction**")
        st.caption("Predict interaction between an RNA sequence and a protein.")
        st.markdown("Input: RNA + protein sequence pairs + labels")
        st.markdown("Output: interaction probability")
        st.warning("Coming Soon")

with col3:
    with st.container(border=True):
        st.markdown("**Protein Function (GO)**")
        st.caption("Predict Gene Ontology terms for a protein from its sequence.")
        st.markdown("Input: protein sequence CSV + GO term labels")
        st.markdown("Output: multilabel GO annotations")
        st.warning("Coming Soon")

    with st.container(border=True):
        st.markdown("**Protein–DNA Interaction**")
        st.caption("Predict whether a protein binds a given DNA sequence motif.")
        st.markdown("Input: protein + DNA sequence pairs + labels")
        st.markdown("Output: binding probability")
        st.warning("Coming Soon")

st.divider()

# ── Shared Infrastructure ─────────────────────────────────────────────────────
st.subheader("Platform Architecture")

st.markdown("""
All task modules share the same training pipeline:

1. **Encode** — sequences are embedded using ESM2 (35M, `esm2_t12_35M_UR50D`)
   via a GPU Celery worker. Long sequences use a sliding window (stride 512).
2. **Represent** — sequence embeddings are combined into a pair vector
   (concatenation + element-wise product + absolute difference).
3. **Train** — a configurable MLP classifier is trained with class-weighted cross-entropy.
   Metrics are written per epoch and polled live by the frontend.
4. **Infer** — the saved model scores new pairs and returns a downloadable CSV.

```
Input CSV  →  ESM2 Encoder (GPU)  →  Pair Representation  →  MLP  →  Prediction
```
""")

st.divider()

st.subheader("Input Format")
st.markdown("""
Training jobs expect a CSV with columns specific to each task.
For PPI (the current available task):

| proteinA | proteinB | label |
|---|---|---|
| MKTAYIAK... | MSHHWGYG... | 1 |
| MNIFEMLR... | MKTAYIAK... | 0 |

Sample files for all tasks will be provided in `examples/` as each module is released.
""")

st.divider()
st.caption("Deep-Prot Studio · Computational Biology & Systems Biology Lab · Research Use Only")
