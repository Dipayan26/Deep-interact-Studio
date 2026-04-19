"""
Inference page — /inference
Run a trained FlexiblePPIModel on new protein pairs, then visualise results
with an 8-tab interactive dashboard:
  ROC & PR Curves · Confusion Matrix · Epoch Curves · KDE · SHAP ·
  Score Scatter · Raw Results
"""

import io
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

BACKEND = os.getenv("BACKEND_URL", "http://backend:8005")
is_dark = st.session_state.get("theme_mode", "Light") == "Dark"
plotly_template = st.session_state.get("plotly_template", "plotly_white")

C_POS   = "#185FA5"
C_NEG   = "#D85A30"
C_GREEN = "#1D9E75"
C_AMBER = "#BA7517"
BG      = "#343a42" if is_dark else "#ffffff"
TXT     = "#e6e8eb" if is_dark else "#1f2937"
SUBTXT  = "#c7ccd3" if is_dark else "#6b7280"

PLOTLY_LAYOUT = dict(
    template=plotly_template,
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="sans-serif", size=12, color=TXT),
    margin=dict(l=50, r=20, t=36, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color=TXT)),
    xaxis=dict(gridcolor="#545c68" if is_dark else "#e5e7eb", zerolinecolor="#545c68" if is_dark else "#e5e7eb"),
    yaxis=dict(gridcolor="#545c68" if is_dark else "#e5e7eb", zerolinecolor="#545c68" if is_dark else "#e5e7eb"),
)

# =============================================================================
# Page header
# =============================================================================

st.title("Inference")
st.caption("Run a trained model on new protein pairs — then explore results interactively.")
st.divider()

# =============================================================================
# 1. Select trained model
# =============================================================================

st.subheader("Select Trained Model")

try:
    r    = requests.get(f"{BACKEND}/jobs", timeout=5)
    jobs = r.json()
    completed_training = [
        j for j in jobs
        if j.get("status") == "completed" and j.get("job_type", "train") == "train"
    ]
except Exception as e:
    st.error(f"Could not load jobs from backend: {e}")
    st.stop()

if not completed_training:
    st.info("No completed training jobs found. Train a model first on the Build page.")
    st.stop()

# ── task-type config ─────────────────────────────────────────────────────────
TASK_INPUT_CONFIGS = {
    "ppi": {
        "required_cols": ["proteinA", "proteinB"],
        "description":   (
            "Upload a CSV with columns **`proteinA`** and **`proteinB`** "
            "(amino acid sequences). "
            "Optionally include a **`label`** column (0/1) to unlock ROC/PR curves "
            "and a confusion matrix."
        ),
        "col_hint": "`proteinA`, `proteinB` · optional: `label`",
    },
    "dti": {
        "required_cols": ["smiles", "sequence"],
        "description":   (
            "Upload a CSV with columns **`smiles`** (SMILES string) and "
            "**`sequence`** (amino acid sequence). "
            "Optionally include a **`label`** column (0/1) for evaluation metrics."
        ),
        "col_hint": "`smiles`, `sequence` · optional: `label`",
    },
    "rpi": {
        "required_cols": ["rna_sequence", "protein_sequence"],
        "description":   (
            "Upload a CSV with columns **`rna_sequence`** (RNA sequence, U or T bases) and "
            "**`protein_sequence`** (amino acid sequence). "
            "Optionally include a **`label`** column (0/1) for evaluation metrics."
        ),
        "col_hint": "`rna_sequence`, `protein_sequence` · optional: `label`",
    },
    "pdi": {
        "required_cols": ["dna_sequence", "protein_sequence"],
        "description":   (
            "Upload a CSV with columns **`dna_sequence`** (DNA sequence) and "
            "**`protein_sequence`** (amino acid sequence). "
            "Optionally include a **`label`** column (0/1) for evaluation metrics."
        ),
        "col_hint": "`dna_sequence`, `protein_sequence` · optional: `label`",
    },
}
DEFAULT_TASK = "ppi"

job_map = {j["run_id"]: j for j in completed_training}

# ── Run ID input with Select / Reset buttons ──────────────────────────────────
st.session_state.setdefault("selected_source_run_id", None)

_confirmed = st.session_state["selected_source_run_id"]

inp_col, btn_col, rst_col = st.columns([5, 1.2, 1])
with inp_col:
    _typed_id = st.text_input(
        "Training Run ID",
        value=_confirmed or "",
        placeholder="e.g. abc123ef",
        label_visibility="collapsed",
        disabled=_confirmed is not None,
    ).strip()

with btn_col:
    if st.button("Select", type="primary", use_container_width=True, disabled=_confirmed is not None):
        if not _typed_id:
            st.error("Enter a Run ID first.")
        elif _typed_id not in job_map:
            st.error(f"`{_typed_id}` not found among completed training jobs.")
        else:
            st.session_state["selected_source_run_id"] = _typed_id
            st.session_state.pop("infer_run_id",    None)
            st.session_state.pop("infer_is_single", None)
            st.rerun()

with rst_col:
    if st.button("Reset", use_container_width=True):
        st.session_state["selected_source_run_id"] = None
        st.session_state.pop("infer_run_id",    None)
        st.session_state.pop("infer_is_single", None)
        st.rerun()

source_run_id = st.session_state.get("selected_source_run_id")

if not source_run_id:
    st.info("Enter a Training Run ID above and click **Select** to continue.")
    st.stop()

if source_run_id not in job_map:
    st.error(f"Run ID `{source_run_id}` no longer found. Click Reset and try again.")
    st.stop()

selected_job   = job_map[source_run_id]
task_type      = selected_job.get("task_type", DEFAULT_TASK)
task_cfg       = TASK_INPUT_CONFIGS.get(task_type, TASK_INPUT_CONFIGS[DEFAULT_TASK])
layer_configs  = selected_job.get("layer_configs", [])
esm_model  = selected_job.get("esm_model", "—")
esm_dim    = selected_job.get("esm_dim") or 480
chem_model = selected_job.get("chem_model", "—")
chem_dim   = selected_job.get("chem_dim") or 384
rna_model  = selected_job.get("rna_model", "—")
rna_dim    = selected_job.get("rna_dim") or 640
dna_model  = selected_job.get("dna_model", "—")
dna_dim    = selected_job.get("dna_dim") or 768

if task_type == "dti":
    input_dim = chem_dim + esm_dim
elif task_type == "rpi":
    input_dim = rna_dim + esm_dim
elif task_type == "pdi":
    input_dim = dna_dim + esm_dim
else:
    input_dim = 2 * esm_dim

st.markdown(
    f"""
    <div style="
        display:inline-block; padding:6px 16px; border-radius:8px;
        background:#1a5fa520; border:1.5px solid #1a5fa560;
        font-size:0.92rem; font-weight:600; color:#1a5fa5; margin-bottom:4px;">
      {task_type.upper()}&nbsp;&nbsp;·&nbsp;&nbsp;
      acc&nbsp;=&nbsp;{selected_job.get('val_acc') or '—'}&nbsp;&nbsp;·&nbsp;&nbsp;
      auroc&nbsp;=&nbsp;{selected_job.get('auroc') or '—'}
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Model details expander ────────────────────────────────────────────────────
def _approx_params(input_dim: int, layer_configs: list) -> int:
    total, cur = 0, input_dim
    for cfg in layer_configs:
        lt = cfg.get("type", "linear").lower()
        if lt == "linear":
            h = int(cfg.get("hidden_dim", 256))
            total += cur * h + h
            if cfg.get("batchnorm"):
                total += 2 * h
            cur = h
        elif lt == "cnn1d":
            out_ch = int(cfg.get("out_channels", 64))
            k      = int(cfg.get("kernel_size", 3))
            total += out_ch * k + out_ch
            cur = out_ch
        elif lt in ("bilstm", "gru"):
            h     = int(cfg.get("hidden_size", 128))
            bidir = 2 if (lt == "bilstm" or cfg.get("bidirectional", True)) else 1
            total += bidir * (4 if lt == "bilstm" else 3) * (cur * h + h * h + h)
            cur = bidir * h
        elif lt == "transformer":
            d  = int(cfg.get("d_model", 256))
            ff = int(cfg.get("dim_feedforward", d * 2))
            nl = int(cfg.get("num_layers", 2))
            total += cur * d + d + nl * (4 * d * d + d * ff + ff * d + 4 * d)
            cur = d
        elif lt == "residual":
            h = int(cfg.get("hidden_dim", 256))
            total += cur * h + h + h * cur + cur + 2 * cur
            if cfg.get("batchnorm"):
                total += 2 * h
    total += cur + 1   # output head
    return total

with st.expander("Model details", expanded=True):
    mc1, mc2, mc3 = st.columns(3)
    n_params = _approx_params(input_dim, layer_configs) if layer_configs else 0
    _card = lambda label, val: f"""
        <div style="padding:4px 0">
            <div style="font-size:0.78rem;color:{SUBTXT};margin-bottom:2px">{label}</div>
            <div style="font-size:0.9rem;font-weight:600">{val}</div>
        </div>"""
    _esm_label = esm_model.replace("esm2_", "ESM2 ").split("_UR")[0]
    if task_type == "dti":
        _emb_str = f"ChemBERTa {chem_dim}-dim + {_esm_label} {esm_dim}-dim"
        _dim_str = f"{input_dim:,} ({chem_dim} chem + {esm_dim} prot)"
    elif task_type == "rpi":
        _rna_label = rna_model.split("/")[-1] if "/" in rna_model else rna_model
        _emb_str = f"RNA-FM `{_rna_label}` {rna_dim}-dim + {_esm_label} {esm_dim}-dim"
        _dim_str = f"{input_dim:,} ({rna_dim} rna + {esm_dim} prot)"
    elif task_type == "pdi":
        _dna_label = dna_model.split("/")[-1] if "/" in dna_model else dna_model
        _emb_str = f"DNABERT `{_dna_label}` {dna_dim}-dim + {_esm_label} {esm_dim}-dim"
        _dim_str = f"{input_dim:,} ({dna_dim} dna + {esm_dim} prot)"
    else:
        _emb_str = _esm_label
        _dim_str = f"{input_dim:,} (2 × {esm_dim})"
    mc1.markdown(_card("Embedding model", _emb_str), unsafe_allow_html=True)
    mc2.markdown(_card("Input dim", _dim_str), unsafe_allow_html=True)
    mc3.markdown(_card("Approx. parameters", f"{n_params:,}" if n_params else "—"), unsafe_allow_html=True)

    if layer_configs:
        rows = []
        cur = input_dim
        for i, cfg in enumerate(layer_configs):
            lt = cfg.get("type", "linear").lower()
            details = {
                "linear":      lambda c: f"hidden={c.get('hidden_dim',256)}, act={c.get('activation','relu')}, drop={c.get('dropout',0.3)}, bn={c.get('batchnorm',False)}",
                "cnn1d":       lambda c: f"out_ch={c.get('out_channels',64)}, kernel={c.get('kernel_size',3)}, act={c.get('activation','relu')}, drop={c.get('dropout',0.3)}",
                "bilstm":      lambda c: f"hidden={c.get('hidden_size',128)}, layers={c.get('num_layers',1)}, drop={c.get('dropout',0.3)}",
                "gru":         lambda c: f"hidden={c.get('hidden_size',128)}, layers={c.get('num_layers',1)}, bidir={c.get('bidirectional',True)}, drop={c.get('dropout',0.3)}",
                "transformer": lambda c: f"d_model={c.get('d_model',256)}, nhead={c.get('nhead',4)}, layers={c.get('num_layers',2)}, ff={c.get('dim_feedforward',512)}, drop={c.get('dropout',0.1)}",
                "residual":    lambda c: f"hidden={c.get('hidden_dim',256)}, act={c.get('activation','relu')}, drop={c.get('dropout',0.3)}, bn={c.get('batchnorm',False)}",
            }
            rows.append({
                "Layer": f"{i + 1}",
                "Type":  lt.upper(),
                "Config": details.get(lt, lambda c: "")(cfg),
            })
        rows.append({"Layer": "Out", "Type": "LINEAR", "Config": "out=1, sigmoid"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Layer configuration not available for this run.")

st.divider()

# =============================================================================
# 2. Upload inference CSV
# =============================================================================

st.subheader("Input Data")
st.markdown(
    "Upload a CSV with columns **`proteinA`** and **`proteinB`**. "
    "Optionally include a **`label`** column (0 / 1) to unlock ROC/PR curves "
    "and a confusion matrix."
)

infer_file = st.file_uploader("Select CSV file", type=["csv"],
                               label_visibility="collapsed")
st.divider()

# =============================================================================
# 3a. Single-pair mode
# =============================================================================

if input_mode == "Single Pair":
    _ik = st.session_state["infer_input_key"]

    if task_type == "dti":
        # ── DTI single pair ──────────────────────────────────────────────────
        sp1, sp2 = st.columns(2)
        with sp1:
            st.markdown("**Compound (SMILES)**")
            raw_smiles = st.text_area(
                "SMILES",
                height=100,
                placeholder="CC(=O)Nc1ccc(O)cc1",
                help="Paste a valid SMILES string for the compound.",
                key=f"dti_smiles_{_ik}",
                label_visibility="collapsed",
            )
        with sp2:
            st.markdown("**Protein Sequence**")
            raw_seq = st.text_area(
                "Sequence",
                height=100,
                placeholder=">ProteinTarget (optional FASTA header)\nMKTAYIAKQ…",
                help="Paste a raw amino acid sequence or a FASTA block.",
                key=f"dti_seq_{_ik}",
                label_visibility="collapsed",
            )

        if st.button("Predict Binding", type="primary", use_container_width=True):
            smiles_val = raw_smiles.strip()
            seq_val    = _parse_seq(raw_seq)
            missing    = []
            if not smiles_val:
                missing.append("SMILES")
            if not seq_val:
                missing.append("protein sequence")
            if missing:
                st.error(f"Required: {', '.join(missing)}.")
            else:
                csv_bytes = f"smiles,sequence\n{smiles_val},{seq_val}\n".encode()
                with st.spinner("Running inference…"):
                    try:
                        r = requests.post(
                            f"{BACKEND}/run_inference/{source_run_id}",
                            files=[("files", ("pair.csv", csv_bytes, "text/csv"))],
                        )
                        r.raise_for_status()
                        data = r.json()
                        if "error" in data:
                            st.error(data["error"])
                        else:
                            st.success(f"Job submitted — Run ID: `{data['run_id']}`")
                            st.session_state["infer_run_id"]    = data["run_id"]
                            st.session_state["infer_is_single"] = True
                    except Exception as e:
                        st.error(f"Submission failed: {e}")

    elif task_type == "rpi":
        # ── RPI single pair ──────────────────────────────────────────────────
        sp1, sp2 = st.columns(2)
        with sp1:
            st.markdown("**RNA Sequence**")
            raw_rna = st.text_area(
                "RNA",
                height=120,
                placeholder="AUGCUUAGCUAG…  (U or T bases accepted)",
                key=f"rpi_rna_{_ik}",
                label_visibility="collapsed",
            )
        with sp2:
            st.markdown("**Protein Sequence**")
            raw_prot = st.text_area(
                "Protein",
                height=120,
                placeholder=">Protein (optional FASTA header)\nMKTAYIAKQ…",
                key=f"rpi_prot_{_ik}",
                label_visibility="collapsed",
            )

        if st.button("Predict Interaction", type="primary", use_container_width=True):
            rna_val  = raw_rna.strip().upper().replace("T", "U")
            prot_val = _parse_seq(raw_prot)
            if not rna_val or not prot_val:
                st.error("Both RNA and protein sequences are required.")
            else:
                csv_bytes = f"rna_sequence,protein_sequence\n{rna_val},{prot_val}\n".encode()
                with st.spinner("Running inference…"):
                    try:
                        r = requests.post(
                            f"{BACKEND}/run_inference/{source_run_id}",
                            files=[("files", ("pair.csv", csv_bytes, "text/csv"))],
                        )
                        r.raise_for_status()
                        data = r.json()
                        if "error" in data:
                            st.error(data["error"])
                        else:
                            st.success(f"Job submitted — Run ID: `{data['run_id']}`")
                            st.session_state["infer_run_id"]    = data["run_id"]
                            st.session_state["infer_is_single"] = True
                    except Exception as e:
                        st.error(f"Submission failed: {e}")

    elif task_type == "pdi":
        # ── PDI single pair ──────────────────────────────────────────────────
        sp1, sp2 = st.columns(2)
        with sp1:
            st.markdown("**DNA Sequence**")
            raw_dna = st.text_area(
                "DNA",
                height=120,
                placeholder="ATGCTTAG…",
                key=f"pdi_dna_{_ik}",
                label_visibility="collapsed",
            )
        with sp2:
            st.markdown("**Protein Sequence**")
            raw_prot = st.text_area(
                "Protein",
                height=120,
                placeholder=">Protein (optional FASTA header)\nMKTAYIAKQ…",
                key=f"pdi_prot_{_ik}",
                label_visibility="collapsed",
            )

        if st.button("Predict Interaction", type="primary", use_container_width=True):
            dna_val  = raw_dna.strip().upper()
            prot_val = _parse_seq(raw_prot)
            if not dna_val or not prot_val:
                st.error("Both DNA and protein sequences are required.")
            else:
                csv_bytes = f"dna_sequence,protein_sequence\n{dna_val},{prot_val}\n".encode()
                with st.spinner("Running inference…"):
                    try:
                        r = requests.post(
                            f"{BACKEND}/run_inference/{source_run_id}",
                            files=[("files", ("pair.csv", csv_bytes, "text/csv"))],
                        )
                        r.raise_for_status()
                        data = r.json()
                        if "error" in data:
                            st.error(data["error"])
                        else:
                            st.success(f"Job submitted — Run ID: `{data['run_id']}`")
                            st.session_state["infer_run_id"]    = data["run_id"]
                            st.session_state["infer_is_single"] = True
                    except Exception as e:
                        st.error(f"Submission failed: {e}")

    else:
        # ── PPI single pair ──────────────────────────────────────────────────
        sp1, sp2 = st.columns(2)
        with sp1:
            raw_a = st.text_area(
                "Protein A",
                height=160,
                placeholder=">ProteinA (optional FASTA header)\nMKTAYIAKQ…",
                help="Paste a raw amino acid sequence or a FASTA block.",
                key=f"seq_a_{_ik}",
            )
        with sp2:
            raw_b = st.text_area(
                "Protein B",
                height=160,
                placeholder=">ProteinB (optional FASTA header)\nMSEQFLAG…",
                help="Paste a raw amino acid sequence or a FASTA block.",
                key=f"seq_b_{_ik}",
            )

        if st.button("Predict Interaction", type="primary", use_container_width=True):
            seq_a = _parse_seq(raw_a)
            seq_b = _parse_seq(raw_b)
            if not seq_a or not seq_b:
                st.error("Both sequences are required.")
            else:
                csv_bytes = f"proteinA,proteinB\n{seq_a},{seq_b}\n".encode()
                with st.spinner("Running inference…"):
                    try:
                        r = requests.post(
                            f"{BACKEND}/run_inference/{source_run_id}",
                            files=[("files", ("pair.csv", csv_bytes, "text/csv"))],
                        )
                        r.raise_for_status()
                        data = r.json()
                        if "error" in data:
                            st.error(data["error"])
                        else:
                            st.success(f"Job submitted — Run ID: `{data['run_id']}`")
                            st.session_state["infer_run_id"]     = data["run_id"]
                            st.session_state["infer_is_single"]  = True
                    except Exception as e:
                        st.error(f"Submission failed: {e}")

# =============================================================================
# 3b. Batch CSV mode
# =============================================================================

if st.button("Run Inference", type="primary", use_container_width=True):
    if infer_file is None:
        st.error("No file selected.")
    else:
        try:
            infer_df = pd.read_csv(io.BytesIO(infer_file.getvalue()))
            missing  = [c for c in ("proteinA", "proteinB") if c not in infer_df.columns]
            if missing:
                st.error(
                    f"CSV is missing required column(s): "
                    f"{', '.join(f'`{c}`' for c in missing)}. "
                    "Rename your columns to `proteinA` and `proteinB`."
                )
                st.stop()
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()

        with st.spinner("Submitting inference job..."):
            try:
                files = [("files", (infer_file.name, infer_file.getvalue(), "text/csv"))]
                r = requests.post(f"{BACKEND}/run_inference/{source_run_id}", files=files)
                r.raise_for_status()
                data = r.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    st.success(f"Inference job submitted — Run ID: `{data['run_id']}`")
                    st.session_state["infer_run_id"] = data["run_id"]
            except Exception as e:
                st.error(f"Submission failed: {e}")

# =============================================================================
# 4. Poll status
# =============================================================================

infer_run_id = st.session_state.get("infer_run_id", "")
if not infer_run_id:
    st.stop()

st.divider()
st.subheader("Results")

rid_input = st.text_input("Inference Run ID", value=infer_run_id)
rid       = rid_input.strip()

col_check, col_auto = st.columns([1, 3])
with col_check:
    check_btn = st.button("Check Status", key="infer_check")
with col_auto:
    auto = st.checkbox("Auto-refresh", value=False, key="infer_auto")

if not (check_btn or auto):
    st.stop()

try:
    sr = requests.get(f"{BACKEND}/check_status/{rid}", timeout=5)
    sd = sr.json()
except Exception as e:
    st.error(f"Could not reach backend: {e}")
    st.stop()

if "error" in sd:
    st.error("Run ID not found.")
    st.stop()

status = sd["status"]
colour = {"completed": "green", "running": "blue",
          "queued": "orange", "failed": "red"}.get(status, "gray")
st.markdown(f"**Status:** :{colour}[{status.capitalize()}]")

if status == "failed":
    st.error(f"Inference failed: {sd.get('result', 'check backend logs')}")
    st.stop()

if status in ("running", "queued"):
    st.info("Inference in progress...")
    if auto:
        time.sleep(4)
        st.rerun()
    st.stop()

# =============================================================================
# 5. Load all data sources
# =============================================================================

# ── inference results CSV ────────────────────────────────────────────────────
resp_csv = requests.get(f"{BACKEND}/download_results/{rid}", stream=True)
if resp_csv.status_code != 200:
    st.warning("Results file not available yet.")
    st.stop()
results_df = pd.read_csv(io.BytesIO(resp_csv.content))

# ── inference metrics (probabilities, labels, aggregate stats) ───────────────
try:
    mr          = requests.get(f"{BACKEND}/inference_metrics/{rid}", timeout=5)
    inf_metrics = mr.json() if mr.ok else {}
except Exception:
    inf_metrics = {}

has_labels = inf_metrics.get("has_labels", False)
probs      = np.array(inf_metrics.get("probabilities",
             results_df["probability"].tolist() if "probability" in results_df else []))
labels     = np.array(inf_metrics.get("labels", [])) if has_labels else None

# ── source training metrics (epoch history) ──────────────────────────────────
try:
    tm        = requests.get(f"{BACKEND}/metrics/{source_run_id}", timeout=5)
    train_met = tm.json() if tm.ok else {}
except Exception:
    train_met = {}

history = train_met.get("history", {})
has_history = bool(history.get("epoch"))

# ── job detail (hyperparams, layer_configs) ──────────────────────────────────
try:
    jd       = requests.get(f"{BACKEND}/job_detail/{source_run_id}", timeout=5)
    job_det  = jd.json() if jd.ok else {}
except Exception:
    job_det  = {}

src_hp       = job_det.get("hyperparams", {})
layer_configs = src_hp.get("layer_configs", [])
esm_dim       = int(src_hp.get("esm_dim", 480))

# =============================================================================
# 6. Summary metric cards
# =============================================================================

n_pairs    = len(results_df)
n_pos_pred = int((results_df.get("prediction", pd.Series(dtype=int)) == 1).sum())
n_neg_pred = n_pairs - n_pos_pred
mean_prob  = float(probs.mean()) if len(probs) else 0.0

# ── Single-pair result card ───────────────────────────────────────────────────
if is_single and n_pairs == 1:
    prob_val = float(probs[0]) if len(probs) else mean_prob
    pred_val = int(results_df["prediction"].iloc[0]) if "prediction" in results_df.columns else int(prob_val >= 0.5)
    if task_type == "dti":
        label    = "Binding" if pred_val == 1 else "Non-Binding"
        prob_lbl = "Binding probability"
    else:
        label    = "Interacting" if pred_val == 1 else "Not Interacting"
        prob_lbl = "Interaction probability"
    colour = C_POS if pred_val == 1 else C_NEG

    st.markdown(
        f"""
        <div style="
            border-radius:12px; padding:28px 36px; margin:12px 0;
            background:linear-gradient(135deg,{colour}18,{colour}08);
            border:2px solid {colour}55; text-align:center;">
          <div style="font-size:2rem; font-weight:700; color:{colour};">{label}</div>
          <div style="font-size:1.1rem; color:{SUBTXT}; margin-top:6px;">
            {prob_lbl}: <strong>{prob_val:.4f}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download result (.csv)",
        data=resp_csv.content,
        file_name=f"ppi_result_{rid}.csv",
        mime="text/csv",
    )

# ── Batch result cards + full dashboard ──────────────────────────────────────
else:
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Pairs scored",    f"{n_pairs:,}")
    mc2.metric("Predicted +",     f"{n_pos_pred:,}")
    mc3.metric("Predicted −",     f"{n_neg_pred:,}")
    mc4.metric("Mean probability", f"{mean_prob:.3f}")

if has_labels and inf_metrics.get("auroc") is not None:
    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUROC",    f"{inf_metrics['auroc']:.4f}")
    m2.metric("AUPRC",    f"{inf_metrics.get('auprc', 0):.4f}")
    m3.metric("F1",       f"{inf_metrics.get('f1', 0):.4f}")
    m4.metric("Accuracy", f"{inf_metrics.get('accuracy', 0):.4f}")
    m5.metric("MCC",      f"{inf_metrics.get('mcc', 0):.4f}")

st.divider()

st.download_button(
    "Download results (.csv)",
    data=resp_csv.content,
    file_name=f"ppi_results_{rid}.csv",
    mime="text/csv",
)

st.divider()

# =============================================================================
# 7. Dashboard tabs
# =============================================================================

ALL_TABS = [
    "Epoch Curves",
    "KDE",
    "SHAP",
    "Probability Distribution",
    "Score Scatter",
    "Raw Results",
]
if has_labels:
    ALL_TABS = ["ROC & PR Curves", "Confusion Matrix"] + ALL_TABS

tabs = st.tabs(ALL_TABS)
tab_offset = 2 if has_labels else 0  # shift for label-gated tabs


# ── helper: KDE ──────────────────────────────────────────────────────────────
def _kde(data: np.ndarray, bw: float = 0.04, n: int = 200):
    xs = np.linspace(0, 1, n)
    ys = np.array([
        np.mean(np.exp(-0.5 * ((x - data) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)))
        for x in xs
    ])
    return xs, ys


# ---------------------------------------------------------------------------
# Tab: ROC & PR  (label-gated)
# ---------------------------------------------------------------------------
if has_labels and labels is not None and len(labels):
    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve,
        confusion_matrix, matthews_corrcoef,
    )
    fpr, tpr, _      = roc_curve(labels, probs)
    roc_auc          = auc(fpr, tpr)
    prec_a, rec_a, _ = precision_recall_curve(labels, probs)
    pr_auc           = auc(rec_a, prec_a)

    with tabs[0]:
        st.markdown("##### ROC & Precision–Recall curves")
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("ROC curve", "Precision–Recall curve"))
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"AUROC = {roc_auc:.3f}",
            line=dict(color=C_POS, width=2),
            fill="tozeroy", fillcolor="rgba(24,95,165,0.09)"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(color=C_NEG, dash="dash", width=1.5)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=rec_a, y=prec_a, mode="lines", name=f"AUPRC = {pr_auc:.3f}",
            line=dict(color=C_GREEN, width=2),
            fill="tozeroy", fillcolor="rgba(29,158,117,0.09)"
        ), row=1, col=2)
        fig.update_xaxes(title_text="False positive rate", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="True positive rate",  range=[0, 1], row=1, col=1)
        fig.update_xaxes(title_text="Recall",    range=[0, 1], row=1, col=2)
        fig.update_yaxes(title_text="Precision", range=[0, 1], row=1, col=2)
        fig.update_layout(**PLOTLY_LAYOUT, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.markdown("##### Confusion matrix & per-class metrics")
        thr_cm = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01, key="cm_thr")
        preds  = (probs >= thr_cm).astype(int)
        cm     = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        cm_fig = go.Figure(go.Heatmap(
            z=[[tp, fp], [fn, tn]],
            x=["Pred Positive", "Pred Negative"],
            y=["True Positive", "True Negative"],
            text=[[str(tp), str(fp)], [str(fn), str(tn)]],
            texttemplate="%{text}", textfont=dict(size=22),
            colorscale=[[0, "#f5f5f3"], [1, C_POS]], showscale=False,
        ))
        cm_fig.update_layout(**PLOTLY_LAYOUT, height=300,
                             xaxis=dict(side="top"), yaxis=dict(autorange="reversed"))
        st.plotly_chart(cm_fig, use_container_width=True)

        prec_v = tp / (tp + fp) if (tp + fp) else 0
        rec_v  = tp / (tp + fn) if (tp + fn) else 0
        spec_v = tn / (tn + fp) if (tn + fp) else 0
        acc_v  = (tp + tn) / len(labels) if len(labels) else 0
        f1_v   = 2 * prec_v * rec_v / (prec_v + rec_v) if (prec_v + rec_v) else 0
        mcc_v  = matthews_corrcoef(labels, preds)
        st.dataframe(pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall / Sensitivity",
                       "Specificity", "F1", "MCC", "AUROC", "AUPRC"],
            "Value":  [f"{v:.4f}" for v in
                       [acc_v, prec_v, rec_v, spec_v, f1_v, mcc_v, roc_auc, pr_auc]],
        }), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Epoch curves  (sourced from training job metrics)
# ---------------------------------------------------------------------------
with tabs[tab_offset + 0]:
    st.markdown("##### Training epoch curves")
    st.caption(
        f"From the source training run `{source_run_id}`. "
        "Shows how loss and accuracy evolved during training."
    )

    if not has_history:
        st.info("No training history available for this model.")
    else:
        epochs     = history.get("epoch", [])
        train_loss = [v for v in history.get("train_loss", []) if v is not None]
        val_loss   = [v for v in history.get("val_loss",   []) if v is not None]
        train_acc  = [v for v in history.get("train_acc",  []) if v is not None]
        val_acc    = [v for v in history.get("val_acc",    []) if v is not None]

        ep_loss = epochs[:len(train_loss)]
        ep_acc  = epochs[:len(train_acc)]

        fig_ep = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Loss per epoch", "Accuracy per epoch"),
        )

        # Loss
        fig_ep.add_trace(go.Scatter(
            x=ep_loss, y=train_loss, mode="lines", name="Train loss",
            line=dict(color=C_POS, width=2),
        ), row=1, col=1)
        fig_ep.add_trace(go.Scatter(
            x=ep_loss[:len(val_loss)], y=val_loss, mode="lines", name="Val loss",
            line=dict(color=C_NEG, width=2, dash="dot"),
        ), row=1, col=1)

        # Accuracy
        fig_ep.add_trace(go.Scatter(
            x=ep_acc, y=train_acc, mode="lines", name="Train acc",
            line=dict(color=C_POS, width=2),
            showlegend=False,
        ), row=1, col=2)
        fig_ep.add_trace(go.Scatter(
            x=ep_acc[:len(val_acc)], y=val_acc, mode="lines", name="Val acc",
            line=dict(color=C_GREEN, width=2, dash="dot"),
            showlegend=False,
        ), row=1, col=2)

        # Early-stop marker
        if train_met.get("early_stopped"):
            stopped_ep = train_met.get("epoch", ep_loss[-1] if ep_loss else None)
            if stopped_ep:
                for col in (1, 2):
                    fig_ep.add_vline(
                        x=stopped_ep, line_dash="dash",
                        line_color=C_AMBER, line_width=1.5,
                        annotation_text="early stop",
                        annotation_font_color=C_AMBER,
                        annotation_position="top left",
                        row=1, col=col,
                    )

        fig_ep.update_xaxes(title_text="Epoch")
        fig_ep.update_yaxes(title_text="Loss",     row=1, col=1)
        fig_ep.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig_ep.update_layout(**PLOTLY_LAYOUT, height=360)
        st.plotly_chart(fig_ep, use_container_width=True)

        # Summary table of best epochs
        if val_loss and val_acc:
            best_loss_ep = ep_loss[int(np.argmin(val_loss))] if val_loss else "—"
            best_acc_ep  = ep_acc[int(np.argmax(val_acc))]   if val_acc  else "—"
            final_loss   = val_loss[-1]
            final_acc    = val_acc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best val loss at epoch", str(best_loss_ep))
            c2.metric("Best val acc at epoch",  str(best_acc_ep))
            c3.metric("Final val loss",  f"{final_loss:.4f}")
            c4.metric("Final val acc",   f"{final_acc:.4f}")


# ---------------------------------------------------------------------------
# Tab: KDE  (kernel density of predicted probabilities)
# ---------------------------------------------------------------------------
with tabs[tab_offset + 1]:
    st.markdown("##### Kernel density estimate — predicted probabilities")
    st.caption(
        "KDE with bandwidth 0.04. When a label column was supplied, "
        "positive and negative class densities are shown separately."
    )

    thr_kde = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01, key="kde_thr")

    fig_kde = go.Figure()

    if has_labels and labels is not None and len(labels):
        neg_p = probs[labels == 0]
        pos_p = probs[labels == 1]
        if len(neg_p):
            xs, ys = _kde(neg_p)
            fig_kde.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=f"Negative (n={len(neg_p)})",
                line=dict(color=C_NEG, width=2),
                fill="tozeroy", fillcolor="rgba(216,90,48,0.12)"
            ))
        if len(pos_p):
            xs, ys = _kde(pos_p)
            fig_kde.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=f"Positive (n={len(pos_p)})",
                line=dict(color=C_POS, width=2),
                fill="tozeroy", fillcolor="rgba(24,95,165,0.12)"
            ))
    else:
        xs, ys = _kde(probs)
        fig_kde.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"All pairs (n={len(probs)})",
            line=dict(color=C_POS, width=2),
            fill="tozeroy", fillcolor="rgba(24,95,165,0.12)"
        ))

    fig_kde.add_vline(
        x=thr_kde, line_dash="dash", line_color=C_AMBER, line_width=2,
        annotation_text=f"thr = {thr_kde:.2f}",
        annotation_position="top right",
        annotation_font_color=C_AMBER,
    )
    fig_kde.update_xaxes(title_text="Predicted probability", range=[0, 1])
    fig_kde.update_yaxes(title_text="Density")
    fig_kde.update_layout(**PLOTLY_LAYOUT, height=360)
    st.plotly_chart(fig_kde, use_container_width=True)

    # Score scatter ───────────────────────────────────────────────────────────
    with tabs[tab_offset + 1]:
        st.markdown("##### Probability scatter — pair index vs. score")
        st.caption("Each point is one pair. Colour = predicted probability. "
                   "When labels are present, filled circle = positive, open = negative.")
        scatter_df = results_df.copy()
        scatter_df["idx"] = range(len(scatter_df))
        if "probability" not in scatter_df.columns:
            scatter_df["probability"] = probs
        if task_type == "dti":
            col_a_name, col_b_name = "smiles", "sequence"
        elif task_type == "rpi":
            col_a_name, col_b_name = "rna_sequence", "protein_sequence"
        elif task_type == "pdi":
            col_a_name, col_b_name = "dna_sequence", "protein_sequence"
        else:
            col_a_name, col_b_name = "proteinA", "proteinB"
        short_a = scatter_df.get(col_a_name, scatter_df.iloc[:, 0]).astype(str).str[:20] + "…"
        short_b = scatter_df.get(col_b_name, scatter_df.iloc[:, 1]).astype(str).str[:20] + "…"
        scatter_df["hover"] = short_a + " × " + short_b
        if has_labels and labels is not None and len(labels) == len(scatter_df):
            scatter_df["true_label"] = labels.astype(int).astype(str)
            fig_sc = px.scatter(scatter_df, x="idx", y="probability",
                color="probability", color_continuous_scale=[C_NEG, C_POS],
                symbol="true_label", symbol_map={"0": "circle-open", "1": "circle"},
                hover_name="hover",
                hover_data={"idx": False, "probability": ":.3f", "true_label": True},
                labels={"idx": "Pair index", "probability": "Probability",
                        "true_label": "True label"})
        else:
            fig_sc = px.scatter(scatter_df, x="idx", y="probability",
                color="probability", color_continuous_scale=[C_NEG, C_POS],
                hover_name="hover",
                hover_data={"idx": False, "probability": ":.3f"},
                labels={"idx": "Pair index", "probability": "Probability"})
        thr_sc = st.slider("Highlight cut-off", 0.0, 1.0, 0.5, 0.01, key="sc_thr")
        fig_sc.add_hline(y=thr_sc, line_dash="dash", line_color=C_AMBER, line_width=1.5)
        fig_sc.update_traces(marker=dict(size=7, opacity=0.75))
        fig_sc.update_layout(**PLOTLY_LAYOUT, height=360,
                             coloraxis_colorbar=dict(title="P(interact)"))
        st.plotly_chart(fig_sc, use_container_width=True)
        n_above = int((scatter_df["probability"] >= thr_sc).sum())
        st.caption(f"{n_above} / {n_pairs} pairs ≥ {thr_sc:.2f}")

    # Raw results table ───────────────────────────────────────────────────────
    with tabs[tab_offset + 2]:
        st.markdown("##### All scored pairs")
        if task_type == "dti":
            search_cols = ["smiles", "sequence"]
            search_ph   = "e.g. CC(=O)… or MKTAY…"
        elif task_type == "rpi":
            search_cols = ["rna_sequence", "protein_sequence"]
            search_ph   = "e.g. AUGCUU… or MKTAY…"
        elif task_type == "pdi":
            search_cols = ["dna_sequence", "protein_sequence"]
            search_ph   = "e.g. ATGCTT… or MKTAY…"
        else:
            search_cols = ["proteinA", "proteinB"]
            search_ph   = "e.g. MKTAY…"
        search  = st.text_input("Filter by substring", placeholder=search_ph)
        show_df = results_df.copy()
        if search.strip():
            mask = pd.Series([False] * len(show_df), index=show_df.index)
            for sc in search_cols:
                if sc in show_df.columns:
                    mask |= show_df[sc].astype(str).str.contains(search, case=False, na=False)
            show_df = show_df[mask]
        for col in search_cols:
            if col in show_df.columns:
                show_df[col] = show_df[col].astype(str).str[:40] + "…"
        st.dataframe(show_df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(show_df):,} of {n_pairs:,} rows")

    st.divider()
    st.markdown("##### Threshold sensitivity")
    thr_rows = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        n = int((results_df.get("probability", pd.Series(probs)) >= t).sum())
        thr_rows.append({"Threshold": t, "Predicted positive": n,
                         "Predicted negative": n_pairs - n})
    st.dataframe(pd.DataFrame(thr_rows), use_container_width=True, hide_index=True)