import streamlit as st

theme_mode = st.session_state.get("theme_mode", "Light")
is_dark = theme_mode == "Dark"

hero_title_color = "#e5e7eb" if is_dark else "#0f172a"
hero_sub_color = "#94a3b8" if is_dark else "#64748b"

badge_av_bg = "#14532d" if is_dark else "#dcfce7"
badge_av_fg = "#bbf7d0" if is_dark else "#15803d"
badge_av_border = "#166534" if is_dark else "#86efac"

badge_soon_bg = "#1f2937" if is_dark else "#f1f5f9"
badge_soon_fg = "#9ca3af" if is_dark else "#94a3b8"
badge_soon_border = "#374151" if is_dark else "#e2e8f0"

st.html("""
<style>
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: """ + hero_title_color + """;
    margin: 0 0 6px;
}
.hero-sub {
    font-size: 1rem;
    color: """ + hero_sub_color + """;
    margin: 0 0 18px;
}
.badge-available {
    display: inline-block;
    background: """ + badge_av_bg + """;
    color: """ + badge_av_fg + """;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 999px;
    border: 1px solid """ + badge_av_border + """;
}
.badge-soon {
    display: inline-block;
    background: """ + badge_soon_bg + """;
    color: """ + badge_soon_fg + """;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 999px;
    border: 1px solid """ + badge_soon_border + """;
}
</style>
""")

# ── Header ────────────────────────────────────────────────────────────────────
st.html("""
<div style="padding: 8px 0 4px;">
  <div class="hero-title">Deep-Prot Studio</div>
  <div class="hero-sub">GPU-accelerated deep learning for biological sequence prediction</div>
</div>
""")

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
        st.html('<span class="badge-available">&#10003;&nbsp; Available</span>')

    with st.container(border=True):
        st.markdown("**Drug-Target Protein Interaction**")
        st.caption("Predict binding between a small molecule (SMILES) and a target protein.")
        st.html('<span class="badge-available">&#10003;&nbsp; Available</span>')

with col2:
    with st.container(border=True):
        st.markdown("**Subcellular Localization**")
        st.caption("Classify where a protein resides within a cell across 10 compartments.")
        st.html('<span class="badge-soon">Coming Soon</span>')

    with st.container(border=True):
        st.markdown("**RNA–Protein Interaction**")
        st.caption("Predict interaction between an RNA sequence and a protein.")
        st.html('<span class="badge-available">&#10003;&nbsp; Available</span>')

with col3:
    with st.container(border=True):
        st.markdown("**Protein Function (GO)**")
        st.caption("Predict Gene Ontology terms for a protein from its sequence.")
        st.html('<span class="badge-soon">Coming Soon</span>')

    with st.container(border=True):
        st.markdown("**Protein–DNA Interaction**")
        st.caption("Predict whether a protein binds a given DNA sequence motif.")
        st.html('<span class="badge-available">&#10003;&nbsp; Available</span>')

st.divider()

# ── Platform Architecture ─────────────────────────────────────────────────────
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
st.caption("Deep-Prot Studio · Computational Biology & Systems Biology Lab · Research Use Only")
