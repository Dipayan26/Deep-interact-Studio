import base64
from pathlib import Path

import streamlit as st


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "home"


def _asset_data_url(filename: str, mime: str) -> str:
    data = (ASSET_DIR / filename).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


hero_img = _asset_data_url("hero_6cfz_transparent.png", "image/png")
ppi_img = _asset_data_url("card_ppi.webp", "image/webp")
dtpi_img = _asset_data_url("card_dtpi.webp", "image/webp")
rpi_img = _asset_data_url("card_rpi.webp", "image/webp")
pdi_img = _asset_data_url("card_pdi.webp", "image/webp")

st.html(
    f"""
<style>
    .block-container {{
        padding-left: 2.2rem;
        padding-right: 2.2rem;
        max-width: 100%;
    }}

    .home-shell {{
        color: #0f172a;
        font-family: "Inter", "Source Sans Pro", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .home-shell * {{
        box-sizing: border-box;
        letter-spacing: 0;
    }}

    .home-hero {{
        position: relative;
        min-height: 470px;
        overflow: hidden;
        padding: clamp(42px, 6vw, 82px) clamp(24px, 5vw, 74px);
        background: #edf4f7;
        border: 1px solid #d9e5ea;
        border-radius: 8px;
        isolation: isolate;
    }}

    .home-hero::after {{
        content: "";
        position: absolute;
        inset: 0;
        background-image: url("{hero_img}");
        background-repeat: no-repeat;
        background-size: min(72vw, 1180px) auto;
        background-position: right -120px bottom -150px;
        opacity: 0.92;
        z-index: -1;
    }}

    .hero-copy {{
        max-width: 760px;
        position: relative;
        z-index: 1;
    }}

    .hero-kicker {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 12px;
        border: 1px solid #b8ccd5;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.72);
        color: #245363;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
    }}

    .hero-title {{
        margin: 22px 0 14px;
        max-width: 720px;
        color: #10212b;
        font-size: clamp(3rem, 7vw, 6.6rem);
        line-height: 0.94;
        font-weight: 850;
    }}

    .hero-subtitle {{
        max-width: 650px;
        margin: 0 0 30px;
        color: #344854;
        font-size: clamp(1.05rem, 1.5vw, 1.35rem);
        line-height: 1.55;
        font-weight: 450;
    }}

    .hero-actions {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
    }}

    .home-pill {{
        display: inline-flex;
        align-items: center;
        min-height: 40px;
        padding: 10px 14px;
        border: 1px solid #c6d5dc;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.82);
        color: #223640;
        font-size: 0.9rem;
        font-weight: 700;
    }}

    .quick-strip {{
        display: grid;
        grid-template-columns: repeat(4, minmax(150px, 1fr));
        gap: 12px;
        margin: 18px 0 42px;
    }}

    .quick-item {{
        min-height: 88px;
        padding: 16px 18px;
        border: 1px solid #dbe5ea;
        border-radius: 8px;
        background: #ffffff;
    }}

    .quick-value {{
        color: #132d38;
        font-size: 1.55rem;
        line-height: 1;
        font-weight: 850;
    }}

    .quick-label {{
        margin-top: 8px;
        color: #60717b;
        font-size: 0.88rem;
        line-height: 1.35;
        font-weight: 600;
    }}

    .section-head {{
        margin: 0 0 20px;
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 18px;
    }}

    .section-title {{
        margin: 0;
        color: #10212b;
        font-size: clamp(1.65rem, 2.2vw, 2.3rem);
        line-height: 1.08;
        font-weight: 820;
    }}

    .section-copy {{
        max-width: 650px;
        margin: 8px 0 0;
        color: #60717b;
        font-size: 1rem;
        line-height: 1.55;
    }}

    .interaction-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(210px, 1fr));
        gap: 16px;
        margin-bottom: 42px;
    }}

    .interaction-card {{
        position: relative;
        min-height: 460px;
        overflow: hidden;
        border: 1px solid #d8e1e6;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.07);
    }}

    .interaction-card img {{
        width: 100%;
        height: 100%;
        min-height: 460px;
        display: block;
        object-fit: cover;
        transition: transform 220ms ease, filter 220ms ease;
    }}

    .interaction-card:hover img {{
        transform: scale(1.035);
        filter: saturate(0.92) brightness(0.96);
    }}

    .card-title {{
        position: absolute;
        left: 12px;
        right: 12px;
        bottom: 12px;
        z-index: 2;
        padding: 12px 14px;
        border: 1px solid rgba(191, 207, 216, 0.95);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.9);
        color: #10212b;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.13);
    }}

    .card-title strong {{
        display: block;
        font-size: 1.08rem;
        line-height: 1.18;
        text-align: center;
        font-weight: 820;
    }}

    .card-overlay {{
        position: absolute;
        inset: 0;
        z-index: 3;
        display: flex;
        flex-direction: column;
        justify-content: end;
        padding: 22px;
        border: 1px solid rgba(188, 204, 213, 0.92);
        background: rgba(248, 251, 252, 0.91);
        color: #10212b;
        opacity: 0;
        transform: translateY(10px);
        transition: opacity 190ms ease, transform 190ms ease;
    }}

    .interaction-card:hover .card-overlay {{
        opacity: 1;
        transform: translateY(0);
    }}

    .card-overlay h3 {{
        margin: 0 0 10px;
        font-size: 1.25rem;
        line-height: 1.18;
        font-weight: 820;
    }}

    .card-overlay p {{
        margin: 0 0 18px;
        color: #3f5561;
        font-size: 0.92rem;
        line-height: 1.48;
    }}

    .card-meta {{
        display: grid;
        gap: 8px;
        color: #2f4652;
        font-size: 0.82rem;
        line-height: 1.35;
    }}

    .card-meta span {{
        display: block;
        padding-left: 10px;
        border-left: 3px solid currentColor;
    }}

    .tone-ppi .card-meta span {{ border-left-color: #0f766e; }}
    .tone-dtpi .card-meta span {{ border-left-color: #2563eb; }}
    .tone-rpi .card-meta span {{ border-left-color: #a21caf; }}
    .tone-pdi .card-meta span {{ border-left-color: #c2410c; }}

    .workflow-band {{
        margin-top: 10px;
        padding: clamp(24px, 3vw, 38px);
        border: 1px solid #d8e1e6;
        border-radius: 8px;
        background: #ffffff;
    }}

    .workflow-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(170px, 1fr));
        gap: 12px;
        margin-top: 20px;
    }}

    .workflow-step {{
        min-height: 150px;
        padding: 18px;
        border: 1px solid #dce7ec;
        border-radius: 8px;
        background: #f8fbfc;
    }}

    .step-number {{
        display: inline-flex;
        width: 34px;
        height: 34px;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: #17333f;
        color: #ffffff;
        font-size: 0.9rem;
        font-weight: 800;
    }}

    .workflow-step h3 {{
        margin: 14px 0 8px;
        color: #17333f;
        font-size: 1.02rem;
        line-height: 1.2;
    }}

    .workflow-step p {{
        margin: 0;
        color: #5f717b;
        font-size: 0.9rem;
        line-height: 1.45;
    }}

    .footer-note {{
        margin: 34px 0 0;
        color: #70818a;
        font-size: 0.82rem;
        text-align: center;
    }}

    @media (max-width: 1180px) {{
        .interaction-grid {{
            grid-template-columns: repeat(2, minmax(240px, 1fr));
        }}

        .workflow-grid,
        .quick-strip {{
            grid-template-columns: repeat(2, minmax(160px, 1fr));
        }}
    }}

    @media (max-width: 760px) {{
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        .home-hero {{
            min-height: 590px;
            padding: 32px 22px;
        }}

        .home-hero::after {{
            background-size: 860px auto;
            background-position: right -210px bottom -70px;
            opacity: 0.72;
        }}

        .hero-title {{
            font-size: clamp(2.55rem, 14vw, 4.4rem);
        }}

        .quick-strip,
        .interaction-grid,
        .workflow-grid {{
            grid-template-columns: 1fr;
        }}

        .section-head {{
            display: block;
        }}

        .interaction-card,
        .interaction-card img {{
            min-height: 390px;
        }}
    }}
</style>

<main class="home-shell">
    <section class="home-hero">
        <div class="hero-copy">
            <div class="hero-kicker">Sequence-based molecular interaction modeling</div>
            <h1 class="hero-title">Deep-Interact Studio</h1>
            <p class="hero-subtitle">
                A webtool for building, training, comparing, and applying deep learning models
                for protein, drug, RNA, and DNA interaction prediction from biological sequence data.
            </p>
            <div class="hero-actions">
                <span class="home-pill">Four interaction builders</span>
                <span class="home-pill">Live training metrics</span>
                <span class="home-pill">Downloadable models and predictions</span>
            </div>
        </div>
    </section>

    <section class="quick-strip" aria-label="Platform summary">
        <div class="quick-item">
            <div class="quick-value">PPI</div>
            <div class="quick-label">Protein pair interaction prediction</div>
        </div>
        <div class="quick-item">
            <div class="quick-value">DTPI</div>
            <div class="quick-label">Drug and target protein binding prediction</div>
        </div>
        <div class="quick-item">
            <div class="quick-value">RPI</div>
            <div class="quick-label">RNA and protein interaction prediction</div>
        </div>
        <div class="quick-item">
            <div class="quick-value">PDI</div>
            <div class="quick-label">Protein and DNA interaction prediction</div>
        </div>
    </section>

    <section>
        <div class="section-head">
            <div>
                <h2 class="section-title">Available Interaction Models</h2>
                <p class="section-copy">
                    Each module turns uploaded molecular pairs into embeddings, trains a configurable
                    classifier, tracks performance during training, and produces reusable prediction artefacts.
                </p>
            </div>
        </div>

        <div class="interaction-grid">
            <article class="interaction-card tone-ppi">
                <img src="{ppi_img}" alt="Protein-protein interaction structure">
                <div class="card-title">
                    <strong>Protein-Protein Interaction</strong>
                </div>
                <div class="card-overlay">
                    <h3>Protein-Protein Interaction</h3>
                    <p>Predict whether two proteins physically interact using sequence-derived protein language model embeddings.</p>
                    <div class="card-meta">
                        <span>Input: protein A sequence and protein B sequence</span>
                        <span>Encoder: ESM2 protein embeddings</span>
                        <span>Output: interaction probability and trained classifier</span>
                    </div>
                </div>
            </article>

            <article class="interaction-card tone-dtpi">
                <img src="{dtpi_img}" alt="Drug-target protein interaction structure">
                <div class="card-title">
                    <strong>Drug-Target Protein Interaction</strong>
                </div>
                <div class="card-overlay">
                    <h3>Drug-Target Protein Interaction</h3>
                    <p>Model potential binding between a compound and a protein target from SMILES and target sequence features.</p>
                    <div class="card-meta">
                        <span>Input: SMILES string and protein sequence</span>
                        <span>Encoder: ChemBERTa plus ESM2</span>
                        <span>Output: binding probability and ranked predictions</span>
                    </div>
                </div>
            </article>

            <article class="interaction-card tone-rpi">
                <img src="{rpi_img}" alt="RNA-protein interaction structure">
                <div class="card-title">
                    <strong>RNA-Protein Interaction</strong>
                </div>
                <div class="card-overlay">
                    <h3>RNA-Protein Interaction</h3>
                    <p>Detect RNA-protein associations by combining RNA language model features with protein embeddings.</p>
                    <div class="card-meta">
                        <span>Input: RNA sequence and protein sequence</span>
                        <span>Encoder: RNA-FM plus ESM2</span>
                        <span>Output: interaction probability for RNA-protein pairs</span>
                    </div>
                </div>
            </article>

            <article class="interaction-card tone-pdi">
                <img src="{pdi_img}" alt="Protein-DNA interaction structure">
                <div class="card-title">
                    <strong>Protein-DNA Interaction</strong>
                </div>
                <div class="card-overlay">
                    <h3>Protein-DNA Interaction</h3>
                    <p>Predict protein-DNA binding events by pairing DNA sequence representations with protein embeddings.</p>
                    <div class="card-meta">
                        <span>Input: DNA sequence and protein sequence</span>
                        <span>Encoder: DNABERT-2 plus ESM2</span>
                        <span>Output: DNA-binding probability and reusable model files</span>
                    </div>
                </div>
            </article>
        </div>
    </section>

    <section class="workflow-band">
        <h2 class="section-title">From Molecular Pairs To Predictions</h2>
        <p class="section-copy">
            Deep-Prot Studio combines task-specific encoders, configurable neural classifiers,
            asynchronous GPU jobs, model comparison, and batch inference in one research workflow.
        </p>
        <div class="workflow-grid">
            <div class="workflow-step">
                <span class="step-number">1</span>
                <h3>Prepare Pairs</h3>
                <p>Upload task-specific CSV data for proteins, compounds, RNA, or DNA.</p>
            </div>
            <div class="workflow-step">
                <span class="step-number">2</span>
                <h3>Build Model</h3>
                <p>Select embeddings, layer structure, dropout, activation, and training settings.</p>
            </div>
            <div class="workflow-step">
                <span class="step-number">3</span>
                <h3>Train And Compare</h3>
                <p>Track loss, accuracy, precision, recall, F1, ROC-AUC, and PR-AUC across runs.</p>
            </div>
            <div class="workflow-step">
                <span class="step-number">4</span>
                <h3>Infer Results</h3>
                <p>Apply trained models to new molecular pairs and export prediction tables.</p>
            </div>
        </div>
    </section>

    <div class="footer-note">Deep-Prot Studio - Computational Biology & Systems Biology Lab - Research Use Only</div>
</main>
"""
)

st.write("")

link_cols = st.columns(4)
with link_cols[0]:
    st.page_link("ppi.py", label="Open PPI", icon=":material/hub:")
with link_cols[1]:
    st.page_link("dtpi.py", label="Open DTPI", icon=":material/medication:")
with link_cols[2]:
    st.page_link("rna_prot.py", label="Open RPI", icon=":material/genetics:")
with link_cols[3]:
    st.page_link("prot_dna.py", label="Open PDI", icon=":material/biotech:")
