import streamlit as st

st.title("Protein Function Prediction")
st.caption("Predict Gene Ontology annotations from protein sequence")

st.warning("This module is under development and will be available in a future release.")

st.divider()

st.markdown("""
Gene Ontology (GO) annotation describes protein function across three domains:
**Biological Process** (what the protein does in the cell), **Molecular Function**
(its biochemical activity), and **Cellular Component** (where it acts).
Only ~1% of known proteins are experimentally annotated; the rest rely on
computational prediction for functional characterisation.

This module trains an **ESM2 + MLP multilabel classifier** to predict GO term
membership from protein sequence, using experimental annotations from
UniProtKB/Swiss-Prot as training labels.
""")

st.divider()

st.subheader("Planned Input Format")
st.markdown("""
| sequence | go_terms |
|---|---|
| MKTAYIAKQR... | GO:0005515;GO:0006355 |
| MSHHWGYGKH... | GO:0004197;GO:0006508;GO:0008270 |

- `sequence` — protein amino acid sequence
- `go_terms` — semicolon-separated GO term IDs (from any or all three ontologies)

Users may filter by ontology (BP / MF / CC) or provide a custom set of GO terms as classes.
""")

st.divider()

st.subheader("Planned Architecture")
st.markdown("""
```
Protein sequence  →  ESM2 encoder  →  480-dim embedding
                                              ↓
                              MLP with sigmoid output (one node per GO term)
                                              ↓
                         GO term probabilities (multilabel, threshold 0.5)
```

Evaluation metrics: micro-AUROC, Fmax, AUPR (standard CAFA metrics).
""")

st.divider()
st.caption("Deep-Prot Studio · Coming Soon")
