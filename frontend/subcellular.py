import streamlit as st

st.title("Subcellular Localization")
st.caption("Classify protein compartment location from sequence")

st.warning("This module is under development and will be available in a future release.")

st.divider()

st.markdown("""
Subcellular localization — where a protein resides within the cell — is tightly coupled
to its function. Mislocalization is implicated in numerous diseases including cancer and
neurodegeneration. Experimental determination via microscopy or fractionation is
low-throughput; sequence-based prediction enables proteome-scale annotation.

This module trains an **ESM2 + MLP multiclass classifier** that maps a protein sequence
to one of 10 standard compartment classes (nucleus, cytoplasm, mitochondria, ER, Golgi,
plasma membrane, lysosome, peroxisome, extracellular, endosome).
""")

st.divider()

st.subheader("Planned Input Format")
st.markdown("""
| sequence | location |
|---|---|
| MKTAYIAKQR... | nucleus |
| MSHHWGYGKH... | mitochondria |
| MNIFEMLRID... | cytoplasm |

Supported compartment labels:
`nucleus`, `cytoplasm`, `mitochondria`, `endoplasmic_reticulum`, `golgi`,
`plasma_membrane`, `lysosome`, `peroxisome`, `extracellular`, `endosome`
""")

st.divider()

st.subheader("Planned Architecture")
st.markdown("""
```
Protein sequence  →  ESM2 encoder  →  480-dim embedding
                                              ↓
                                   MLP classifier (10-class softmax)
                                              ↓
                              Compartment label + confidence score
```

Single-sequence task — no pair representation step required.
""")

st.divider()
st.caption("Deep-Prot Studio · Coming Soon")
