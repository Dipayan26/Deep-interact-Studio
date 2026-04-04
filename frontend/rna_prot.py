import streamlit as st

st.title("RNA–Protein Interaction")
st.caption("Predict interaction between RNA sequences and RNA-binding proteins")

st.warning("This module is under development and will be available in a future release.")

st.divider()

st.markdown("""
RNA-binding proteins (RBPs) regulate post-transcriptional gene expression — including
splicing, polyadenylation, translation, and RNA stability. Identifying which RBPs bind
which RNA sequences is fundamental to understanding gene regulation and RNA-related disease.

This module combines **ESM2 protein embeddings** with **RNA-FM nucleotide embeddings**
to predict whether a given RNA sequence is bound by a specific protein, trained on
eCLIP or CLIP-seq derived interaction data.
""")

st.divider()

st.subheader("Planned Input Format")
st.markdown("""
| rnaSequence | proteinSequence | label |
|---|---|---|
| AUGCUAGCUAGCUA... | MKTAYIAKQR... | 1 |
| GCUAGCUAGCUAGC... | MSHHWGYGKH... | 0 |

- `rnaSequence` — RNA sequence (A, U, G, C; up to 512 nucleotides)
- `proteinSequence` — RNA-binding protein amino acid sequence
- `label` — `1` (interacts) or `0` (does not interact)
""")

st.divider()

st.subheader("Planned Architecture")
st.markdown("""
```
RNA sequence     →  RNA-FM encoder   →  RNA embedding (640-dim)
                                                  ↓
Protein sequence →  ESM2 encoder     →  protein embedding (480-dim)
                                                  ↓
                         Concatenate + MLP classifier
                                                  ↓
                              Interaction probability [0, 1]
```

RNA Encoder: RNA-FM (HuggingFace `multimolecule/rnafm`)
""")

st.divider()
st.caption("Deep-Prot Studio · Coming Soon")
