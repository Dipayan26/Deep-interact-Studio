import streamlit as st

st.title("References")
st.caption("Key publications and resources underlying Deep-Prot Studio.")
st.divider()

st.subheader("Protein Language Models")

st.markdown("""
1. **Lin Z, et al.** (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model.
   *Science*, 379(6637), 1123–1130. https://doi.org/10.1126/science.ade2574

2. **Rives A, et al.** (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.
   *PNAS*, 118(15), e2016239118. https://doi.org/10.1073/pnas.2016239118
""")

st.subheader("Protein–Protein Interaction Prediction")

st.markdown("""
3. **Sledzieski S, et al.** (2021). D-SCRIPT translates genome to phenome with sequence-based, structure-aware, genome-scale predictions of protein-protein interactions.
   *Cell Systems*, 12(10), 969–982. https://doi.org/10.1016/j.cels.2021.08.010

4. **Chen M, et al.** (2019). Multifaceted protein–protein interaction prediction based on Siamese residual RCNN.
   *Bioinformatics*, 35(14), i305–i314. https://doi.org/10.1093/bioinformatics/btz328
""")

st.subheader("Tools & Frameworks")

st.markdown("""
5. **Paszke A, et al.** (2019). PyTorch: An imperative style, high-performance deep learning library.
   *NeurIPS 32*. https://pytorch.org

6. **Streamlit** — https://streamlit.io

7. **FastAPI** — https://fastapi.tiangolo.com
""")

st.divider()
st.info("Additional references will be added as new tasks are implemented.")
