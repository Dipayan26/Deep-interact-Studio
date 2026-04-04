import streamlit as st

st.title("Protein–DNA Interaction")
st.caption("Predict binding between transcription factors and DNA sequence motifs")

st.warning("This module is under development and will be available in a future release.")

st.divider()

st.markdown("""
Transcription factors (TFs) and other DNA-binding proteins regulate gene expression
by recognising specific DNA sequence motifs. Predicting these binding events from
sequence is central to understanding gene regulatory networks, identifying
cis-regulatory elements, and modelling transcriptional control.

This module combines **ESM2 protein embeddings** with **DNABERT-2 nucleotide embeddings**
to predict whether a given protein binds a specific DNA sequence, trained on
ChIP-seq and SELEX-derived binding data.
""")

st.divider()

st.subheader("Planned Input Format")
st.markdown("""
| dnaSequence | proteinSequence | label |
|---|---|---|
| ATCGATCGATCG... | MKTAYIAKQR... | 1 |
| GCTAGCTAGCTA... | MSHHWGYGKH... | 0 |

- `dnaSequence` — DNA sequence (A, T, G, C; up to 512 bp)
- `proteinSequence` — DNA-binding protein amino acid sequence
- `label` — `1` (binds) or `0` (does not bind)
""")

st.divider()

st.subheader("Planned Architecture")
st.markdown("""
```
DNA sequence     →  DNABERT-2 encoder  →  DNA embedding (768-dim)
                                                   ↓
Protein sequence →  ESM2 encoder       →  protein embedding (480-dim)
                                                   ↓
                          Concatenate + MLP classifier
                                                   ↓
                               Binding probability [0, 1]
```

DNA Encoder: DNABERT-2 (HuggingFace `zhihan1996/DNABERT-2-117M`)
""")

st.divider()
st.caption("Deep-Prot Studio · Coming Soon")
