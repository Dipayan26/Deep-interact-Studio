import streamlit as st

st.title("Drug–Target Interaction")
st.caption("Predict binding affinity between small molecules and target proteins")

st.warning("This module is under development and will be available in a future release.")

st.divider()

st.markdown("""
Drug–Target Interaction (DTI) prediction identifies whether a small molecule compound
will bind to a specific protein target. Accurate DTI prediction is central to early-stage
drug discovery — it narrows the chemical search space and prioritises compounds for
experimental validation before costly wet-lab screens.

This module will combine **ESM2 protein embeddings** with **ChemBERTa molecular embeddings**
(SMILES-based) to model both sides of the interaction, then train an MLP classifier on
labelled binding data.
""")

st.divider()

st.subheader("Planned Input Format")
st.markdown("""
| smiles | proteinSequence | label |
|---|---|---|
| CC(=O)Oc1ccccc1C(=O)O | MKTAYIAKQR... | 1 |
| c1ccc2c(c1)ccc(=O)o2 | MSHHWGYGKH... | 0 |

- `smiles` — canonical SMILES string of the compound
- `proteinSequence` — target protein amino acid sequence
- `label` — `1` (binds) or `0` (does not bind)
""")

st.divider()

st.subheader("Planned Architecture")
st.markdown("""
```
SMILES  →  ChemBERTa encoder  →  compound embedding (384-dim)
                                           ↓
Protein →  ESM2 encoder       →  protein embedding (480-dim)
                                           ↓
                         Concatenate + MLP classifier
                                           ↓
                              Binding probability [0, 1]
```

Encoder: ChemBERTa-77M-MTR (HuggingFace `seyonec/ChemBERTa-zinc-base-v1`)
""")

st.divider()
st.caption("Deep-Prot Studio · Coming Soon")
