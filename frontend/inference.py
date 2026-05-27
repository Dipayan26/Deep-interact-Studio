"""
Inference page — submit inference jobs against a completed training run.
Results are viewed on the Inference Results page.
"""

import io
import os
import re

import pandas as pd
import requests
import streamlit as st

from model_details import approx_params_from_hp, input_dim_from_hp

BACKEND = os.getenv("BACKEND_URL", "http://backend:8005")
if "infer_input_key" not in st.session_state:
    st.session_state["infer_input_key"] = 0
is_dark = st.session_state.get("theme_mode", "Light") == "Dark"
plotly_template = st.session_state.get("plotly_template", "plotly_white")

C_POS   = "#185FA5"
C_NEG   = "#D85A30"
C_GREEN = "#1D9E75"
C_AMBER = "#BA7517"
BG      = "#343a42" if is_dark else "#ffffff"
TXT     = "#e6e8eb" if is_dark else "#1f2937"
SUBTXT  = "#c7ccd3" if is_dark else "#6b7280"
MAX_SINGLE_PAIR_INPUT_LEN = 512
MAX_BATCH_INFERENCE_PAIRS = 60_000
_BATCH_EDITED_DF_KEY = "infer_batch_edited_df"
_BATCH_EDITED_SIG_KEY = "infer_batch_edited_signature"
VALID_AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBJOUXZ*\-]+$")
VALID_RNA = re.compile(r"^[AUGCNaugcn]+$")
VALID_DNA = re.compile(r"^[ATGCNatgcn]+$")
VALID_SMILES = re.compile(r"^[A-Za-z0-9@+\-\[\]()\=\#\%\\/\.\*~:]+$")

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
st.markdown(
    "**Choose a completed training run, upload compatible input pairs, and submit a new inference job. "
    "Use Job Status to find the training run ID and follow the inference job after submission.**"
)
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
    "dtpi": {
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
            st.session_state.pop("infer_run_id", None)
            st.session_state.pop("single_infer_result", None)
            st.rerun()

with rst_col:
    if st.button("Reset", use_container_width=True):
        st.session_state["selected_source_run_id"] = None
        st.session_state.pop("infer_run_id",    None)
        st.session_state.pop("infer_is_single", None)
        st.session_state.pop("single_infer_result", None)
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
model_hp       = {**selected_job, "task_type": task_type, "layer_configs": layer_configs}
esm_model  = selected_job.get("esm_model", "—")
esm_dim    = selected_job.get("esm_dim") or 480
chem_model = selected_job.get("chem_model", "—")
chem_dim   = selected_job.get("chem_dim") or 384
rna_model  = selected_job.get("rna_model", "—")
rna_dim    = selected_job.get("rna_dim") or 640
dna_model  = selected_job.get("dna_model", "—")
dna_dim    = selected_job.get("dna_dim") or 768

input_dim = input_dim_from_hp(model_hp, task_type)

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

with st.expander("Model details", expanded=True):
    mc1, mc2, mc3, mc4 = st.columns(4)
    n_params = approx_params_from_hp(model_hp, task_type) or 0
    actual_params = selected_job.get("trainable_params")
    _card = lambda label, val: f"""
        <div style="padding:4px 0">
            <div style="font-size:0.78rem;color:{SUBTXT};margin-bottom:2px">{label}</div>
            <div style="font-size:0.9rem;font-weight:600">{val}</div>
        </div>"""
    _esm_label = esm_model.replace("esm2_", "ESM2 ").split("_UR")[0]
    if task_type == "dtpi":
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
    mc4.markdown(_card("Actual parameters", f"{int(actual_params):,}" if actual_params else "—"), unsafe_allow_html=True)

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

st.divider()

# =============================================================================
# 3a. Single-pair mode
# =============================================================================
def _parse_seq(raw: str) -> str:
    if not raw:
        return ""

    lines = raw.strip().splitlines()

    if lines and lines[0].startswith(">"):
        lines = lines[1:]

    seq = "".join(lines).replace(" ", "").upper()

    # Keep only valid amino acid characters
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    seq = "".join([c for c in seq if c in valid])

    return seq


def _run_single_pair(payload: dict):
    st.session_state.pop("infer_run_id", None)
    st.session_state.pop("infer_result_run_id", None)

    progress = st.progress(0)
    status = st.empty()
    status.caption("Preparing input...")
    progress.progress(15)

    try:
        status.caption("Embedding and scoring pair...")
        progress.progress(45)
        r = requests.post(
            f"{BACKEND}/run_single_inference/{source_run_id}",
            json=payload,
        )
        progress.progress(85)
        try:
            data = r.json()
        except Exception:
            data = {}
        if not r.ok:
            st.error(data.get("error", f"Prediction failed with status {r.status_code}."))
            progress.empty()
            status.empty()
            return
        if "error" in data:
            st.error(data["error"])
            progress.empty()
            status.empty()
            return
        st.session_state["single_infer_result"] = data
        progress.progress(100)
        status.caption("Prediction complete.")
    except Exception as e:
        progress.empty()
        status.empty()
        st.error(f"Prediction failed: {e}")


def _length_errors(fields: dict) -> list[str]:
    return [
        f"{label} is {len(str(value)):,} characters; maximum is {MAX_SINGLE_PAIR_INPUT_LEN}."
        for label, value in fields.items()
        if len(str(value)) > MAX_SINGLE_PAIR_INPUT_LEN
    ]


def _show_length_errors(errors: list[str]):
    for err in errors:
        st.error(err)


def _render_single_pair_result():
    data = st.session_state.get("single_infer_result")
    if not data:
        return

    result = data.get("result", {})
    note = result.get("note", "")
    prob = result.get("probability")
    pred = result.get("prediction")

    st.divider()
    st.markdown("**Single Pair Result**")

    if prob is None or pred is None:
        st.warning(note or "Prediction was not available for this pair.")
    else:
        pred = int(pred)
        prob = float(prob)
        label = "Interacting" if pred == 1 else "Not Interacting"
        card_col = C_POS if pred == 1 else C_NEG
        st.markdown(
            f"""
            <div style="border-radius:12px; padding:28px 36px; margin:12px 0;
                background:linear-gradient(135deg,{card_col}18,{card_col}08);
                border:2px solid {card_col}55; text-align:center;">
              <div style="font-size:2rem; font-weight:700; color:{card_col};">{label}</div>
              <div style="font-size:1.1rem; color:{SUBTXT}; margin-top:6px;">
                Interaction probability: <strong>{prob:.4f}</strong>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if note:
            st.caption(note)

    csv_bytes = pd.DataFrame([result]).to_csv(index=False).encode()
    st.download_button(
        "Download result (.csv)",
        data=csv_bytes,
        file_name="single_pair_inference_result.csv",
        mime="text/csv",
        use_container_width=True,
    )


input_mode = st.radio(
    "Input Mode",
    ["Batch CSV", "Single Pair"],
    horizontal=True
)

if input_mode == "Single Pair":
    _ik = st.session_state["infer_input_key"]

    def _clear_single_pair_inputs():
        st.session_state["infer_input_key"] += 1
        st.session_state.pop("single_infer_result", None)
        st.session_state.pop("infer_run_id", None)
        st.session_state.pop("infer_result_run_id", None)
        st.rerun()

    if task_type == "dtpi":
        # ── DTPI single pair ──────────────────────────────────────────────────
        sp1, sp2 = st.columns(2)
        with sp1:
            st.markdown("**Compound (SMILES)**")
            raw_smiles = st.text_area(
                "SMILES",
                height=100,
                placeholder="CC(=O)Nc1ccc(O)cc1",
                help="Paste a valid SMILES string for the compound.",
                key=f"dtpi_smiles_{_ik}",
                label_visibility="collapsed",
            )
        with sp2:
            st.markdown("**Protein Sequence**")
            raw_seq = st.text_area(
                "Sequence",
                height=100,
                placeholder=">ProteinTarget (optional FASTA header)\nMKTAYIAKQ…",
                help="Paste a raw amino acid sequence or a FASTA block.",
                key=f"dtpi_seq_{_ik}",
                label_visibility="collapsed",
            )

        smiles_val = raw_smiles.strip()
        seq_val = _parse_seq(raw_seq)
        length_errors = _length_errors({"SMILES": smiles_val, "protein sequence": seq_val})
        _show_length_errors(length_errors)

        predict_col, clear_col = st.columns([3, 1])
        predict_clicked = predict_col.button(
            "Predict Binding", type="primary", use_container_width=True, disabled=bool(length_errors)
        )
        if clear_col.button("Clear", use_container_width=True, key="clear_single_dtpi"):
            _clear_single_pair_inputs()

        if predict_clicked:
            missing    = []
            if not smiles_val:
                missing.append("SMILES")
            if not seq_val:
                missing.append("protein sequence")
            if missing:
                st.error(f"Required: {', '.join(missing)}.")
            else:
                _run_single_pair({"smiles": smiles_val, "sequence": seq_val})

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

        rna_val = raw_rna.strip().upper().replace("T", "U")
        prot_val = _parse_seq(raw_prot)
        length_errors = _length_errors({"RNA sequence": rna_val, "protein sequence": prot_val})
        _show_length_errors(length_errors)

        predict_col, clear_col = st.columns([3, 1])
        predict_clicked = predict_col.button(
            "Predict Interaction", type="primary", use_container_width=True, disabled=bool(length_errors)
        )
        if clear_col.button("Clear", use_container_width=True, key="clear_single_rpi"):
            _clear_single_pair_inputs()

        if predict_clicked:
            if not rna_val or not prot_val:
                st.error("Both RNA and protein sequences are required.")
            else:
                _run_single_pair({"rna_sequence": rna_val, "protein_sequence": prot_val})

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

        dna_val = raw_dna.strip().upper()
        prot_val = _parse_seq(raw_prot)
        length_errors = _length_errors({"DNA sequence": dna_val, "protein sequence": prot_val})
        _show_length_errors(length_errors)

        predict_col, clear_col = st.columns([3, 1])
        predict_clicked = predict_col.button(
            "Predict Interaction", type="primary", use_container_width=True, disabled=bool(length_errors)
        )
        if clear_col.button("Clear", use_container_width=True, key="clear_single_pdi"):
            _clear_single_pair_inputs()

        if predict_clicked:
            if not dna_val or not prot_val:
                st.error("Both DNA and protein sequences are required.")
            else:
                _run_single_pair({"dna_sequence": dna_val, "protein_sequence": prot_val})

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

        seq_a = _parse_seq(raw_a)
        seq_b = _parse_seq(raw_b)
        length_errors = _length_errors({"Protein A": seq_a, "Protein B": seq_b})
        _show_length_errors(length_errors)

        predict_col, clear_col = st.columns([3, 1])
        predict_clicked = predict_col.button(
            "Predict Interaction", type="primary", use_container_width=True, disabled=bool(length_errors)
        )
        if clear_col.button("Clear", use_container_width=True, key="clear_single_ppi"):
            _clear_single_pair_inputs()

        if predict_clicked:
            if not seq_a or not seq_b:
                st.error("Both sequences are required.")
            else:
                _run_single_pair({"proteinA": seq_a, "proteinB": seq_b})

    _render_single_pair_result()
    st.stop()

# =============================================================================
# 3b. Batch CSV mode  — upload + column mapping + preview + submit
# =============================================================================

if input_mode == "Batch CSV":
    st.markdown("Upload a CSV file. Use **Column Mapping** below to assign your "
                "columns to the required fields — no renaming needed beforehand.")

    batch_file = st.file_uploader(
        "Select CSV file", type=["csv"], label_visibility="collapsed",
        key="infer_file_upload",
    )

    raw_infer_df = None
    if batch_file is not None:
        try:
            raw_infer_df = pd.read_csv(io.BytesIO(batch_file.getvalue()))
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

    if raw_infer_df is not None:
        csv_cols = raw_infer_df.columns.tolist()
        st.caption(
            f"Loaded **{len(raw_infer_df):,} rows** · {len(csv_cols)} columns: "
            f"`{'`, `'.join(csv_cols)}`"
        )

        st.divider()
        st.markdown("**Column Mapping** (★ = required)")

        def _best_guess(candidates: list, cols: list) -> str:
            for c in candidates:
                for col in cols:
                    if c.lower() in col.lower() or col.lower() in c.lower():
                        return col
            return cols[0]

        def _label_select(cols: list, key: str) -> str:
            opts = ["(none)"] + cols
            default = next(
                (i + 1 for i, c in enumerate(cols) if "label" in c.lower()), 0
            )
            return st.selectbox("Label (0/1) — optional", opts,
                                index=default, key=key)

        def _invalid_mask(df: pd.DataFrame, validators: dict[str, callable]) -> pd.Series:
            mask = pd.Series(False, index=df.index)
            for col, validator in validators.items():
                values = df[col].astype(str).str.strip()
                mask |= values.eq("") | values.apply(lambda value: not validator(value))
            return mask

        def _long_sequence_mask(df: pd.DataFrame, length_caps: dict[str, int]) -> pd.Series:
            mask = pd.Series(False, index=df.index)
            for col, max_len in length_caps.items():
                lengths = df[col].astype(str).str.strip().str.len()
                mask |= lengths.gt(max_len)
            return mask

        def _trim_batch_length_columns(df: pd.DataFrame, length_caps: dict[str, int]) -> pd.DataFrame:
            trimmed = df.copy()
            for col, max_len in length_caps.items():
                values = trimmed[col].astype(str).str.strip().str.slice(0, max_len)
                if col in {"proteinA", "proteinB", "sequence", "protein_sequence", "dna_sequence"}:
                    values = values.str.upper()
                elif col == "rna_sequence":
                    values = values.str.upper().str.replace("T", "U", regex=False)
                trimmed[col] = values
            return trimmed.reset_index(drop=True)

        def _render_batch_validation_errors(df: pd.DataFrame, batch_signature: tuple) -> bool:
            blocked = False
            if len(df) > MAX_BATCH_INFERENCE_PAIRS:
                st.error(
                    f"Max inference allowed is {MAX_BATCH_INFERENCE_PAIRS:,} pairs. "
                    f"Your mapped CSV contains {len(df):,} pairs."
                )
                blocked = True

            validators_by_task = {
                "ppi": {
                    "proteinA": lambda value: bool(VALID_AA.match(value.upper())),
                    "proteinB": lambda value: bool(VALID_AA.match(value.upper())),
                },
                "dtpi": {
                    "smiles": lambda value: bool(VALID_SMILES.match(value)) and len(value) > 0,
                    "sequence": lambda value: bool(VALID_AA.match(value.upper())),
                },
                "rpi": {
                    "rna_sequence": lambda value: bool(VALID_RNA.match(value.upper().replace("T", "U"))),
                    "protein_sequence": lambda value: bool(VALID_AA.match(value.upper())),
                },
                "pdi": {
                    "dna_sequence": lambda value: bool(VALID_DNA.match(value.upper())),
                    "protein_sequence": lambda value: bool(VALID_AA.match(value.upper())),
                },
            }
            length_caps_by_task = {
                "ppi": {
                    "proteinA": MAX_SINGLE_PAIR_INPUT_LEN,
                    "proteinB": MAX_SINGLE_PAIR_INPUT_LEN,
                },
                "dtpi": {
                    "smiles": MAX_SINGLE_PAIR_INPUT_LEN,
                    "sequence": MAX_SINGLE_PAIR_INPUT_LEN,
                },
                "rpi": {
                    "rna_sequence": MAX_SINGLE_PAIR_INPUT_LEN,
                    "protein_sequence": MAX_SINGLE_PAIR_INPUT_LEN,
                },
                "pdi": {
                    "dna_sequence": MAX_SINGLE_PAIR_INPUT_LEN,
                    "protein_sequence": MAX_SINGLE_PAIR_INPUT_LEN,
                },
            }
            length_labels_by_task = {
                "ppi": {
                    "proteinA": "Protein A",
                    "proteinB": "Protein B",
                },
                "dtpi": {
                    "smiles": "SMILES",
                    "sequence": "protein sequence",
                },
                "rpi": {
                    "rna_sequence": "RNA sequence",
                    "protein_sequence": "protein sequence",
                },
                "pdi": {
                    "dna_sequence": "DNA sequence",
                    "protein_sequence": "protein sequence",
                },
            }
            type_labels = {
                "ppi": "both mapped columns must be protein sequences",
                "dtpi": "mapped columns must be SMILES and protein sequence",
                "rpi": "mapped columns must be RNA sequence and protein sequence",
                "pdi": "mapped columns must be DNA sequence and protein sequence",
            }
            validators = validators_by_task.get(task_type, {})
            if validators:
                invalid = _invalid_mask(df, validators)
                invalid_count = int(invalid.sum())
                if invalid_count:
                    st.error(
                        f"{invalid_count:,} mapped row(s) do not match the expected input type: "
                        f"{type_labels.get(task_type, 'mapped columns are invalid')}. "
                        "Fix the CSV or choose different columns before running inference."
                    )
                    preview = df.loc[invalid].head(5).copy()
                    preview.insert(0, "source row", preview.index.to_series().astype(int) + 2)
                    for col in preview.select_dtypes(include="object").columns:
                        preview[col] = preview[col].astype(str).str[:50] + "..."
                    st.dataframe(preview, use_container_width=True, hide_index=True)
                    blocked = True

            length_caps = length_caps_by_task.get(task_type, {})
            if length_caps:
                long_mask = _long_sequence_mask(df, length_caps)
                long_count = int(long_mask.sum())
                if long_count:
                    labels = length_labels_by_task.get(task_type, {})
                    max_lengths = {
                        labels.get(col, col): int(df[col].astype(str).str.strip().str.len().max())
                        for col in length_caps
                    }
                    max_details = ", ".join(
                        f"{label} max {max_len:,}" for label, max_len in max_lengths.items()
                    )
                    st.error(
                        f"{long_count:,} mapped row(s) exceed the {MAX_SINGLE_PAIR_INPUT_LEN:,}-character "
                        f"batch inference sequence cap. {max_details}. "
                        "Trim or remove long entries before running inference."
                    )
                    preview = df.loc[long_mask].head(5).copy()
                    preview.insert(0, "source row", preview.index.to_series().astype(int) + 2)
                    for col in length_caps:
                        label = labels.get(col, col)
                        preview[f"{label} length"] = (
                            df[col].astype(str).str.strip().str.len().loc[long_mask].head(5).to_numpy()
                        )
                    for col in preview.select_dtypes(include="object").columns:
                        preview[col] = preview[col].astype(str).str[:50] + "..."
                    st.dataframe(preview, use_container_width=True, hide_index=True)
                    fix_cols = st.columns(2)
                    with fix_cols[0]:
                        if st.button(
                            f"Trim long entries to {MAX_SINGLE_PAIR_INPUT_LEN}",
                            key="infer_trim_long_sequences",
                            use_container_width=True,
                        ):
                            st.session_state[_BATCH_EDITED_DF_KEY] = _trim_batch_length_columns(df, length_caps)
                            st.session_state[_BATCH_EDITED_SIG_KEY] = batch_signature
                            st.rerun()
                    with fix_cols[1]:
                        if st.button(
                            f"Remove {long_count:,} long-sequence row(s)",
                            key="infer_remove_long_sequence_rows",
                            use_container_width=True,
                        ):
                            cleaned = df.loc[~long_mask].copy().reset_index(drop=True)
                            if cleaned.empty:
                                st.error("Removing long-sequence rows would remove all rows. Trim the entries or upload a shorter CSV.")
                            else:
                                st.session_state[_BATCH_EDITED_DF_KEY] = cleaned
                                st.session_state[_BATCH_EDITED_SIG_KEY] = batch_signature
                                st.rerun()
                    blocked = True

            return blocked

        send_df    = None
        mapping_ok = True

        if task_type == "ppi":
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                map_a = st.selectbox("Protein A ★", csv_cols, key="imap_pA",
                    index=csv_cols.index(_best_guess(
                        ["proteinA","protein_a","seqA","seq_a","sequenceA"], csv_cols)))
            with mc2:
                map_b = st.selectbox("Protein B ★", csv_cols, key="imap_pB",
                    index=csv_cols.index(_best_guess(
                        ["proteinB","protein_b","seqB","seq_b","sequenceB"], csv_cols)))
            with mc3:
                map_lbl = _label_select(csv_cols, "imap_lbl")
            if map_a == map_b:
                st.error("Protein A and Protein B must be different columns.")
                mapping_ok = False
            else:
                keep = [map_a, map_b]
                rmap = {map_a: "proteinA", map_b: "proteinB"}
                if map_lbl != "(none)":
                    keep.append(map_lbl); rmap[map_lbl] = "label"
                send_df = raw_infer_df[keep].rename(columns=rmap)

        elif task_type == "dtpi":
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                map_smi = st.selectbox("SMILES (compound) ★", csv_cols, key="imap_smi",
                    index=csv_cols.index(_best_guess(
                        ["smiles","smile","compound","drug","molecule"], csv_cols)))
            with mc2:
                map_seq = st.selectbox("Protein sequence ★", csv_cols, key="imap_seq",
                    index=csv_cols.index(_best_guess(
                        ["sequence","protein","target","seq"], csv_cols)))
            with mc3:
                map_lbl = _label_select(csv_cols, "imap_lbl")
            if map_smi == map_seq:
                st.error("SMILES and protein sequence must be different columns.")
                mapping_ok = False
            else:
                keep = [map_smi, map_seq]
                rmap = {map_smi: "smiles", map_seq: "sequence"}
                if map_lbl != "(none)":
                    keep.append(map_lbl); rmap[map_lbl] = "label"
                send_df = raw_infer_df[keep].rename(columns=rmap)

        elif task_type == "rpi":
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                map_rna = st.selectbox("RNA sequence ★", csv_cols, key="imap_rna",
                    index=csv_cols.index(_best_guess(
                        ["rna","rna_seq","rna_sequence","ncrna"], csv_cols)))
            with mc2:
                map_prot = st.selectbox("Protein sequence ★", csv_cols, key="imap_prot",
                    index=csv_cols.index(_best_guess(
                        ["protein","protein_seq","protein_sequence","seq"], csv_cols)))
            with mc3:
                map_lbl = _label_select(csv_cols, "imap_lbl")
            if map_rna == map_prot:
                st.error("RNA and protein sequence must be different columns.")
                mapping_ok = False
            else:
                keep = [map_rna, map_prot]
                rmap = {map_rna: "rna_sequence", map_prot: "protein_sequence"}
                if map_lbl != "(none)":
                    keep.append(map_lbl); rmap[map_lbl] = "label"
                send_df = raw_infer_df[keep].rename(columns=rmap)

        elif task_type == "pdi":
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                map_dna = st.selectbox("DNA sequence ★", csv_cols, key="imap_dna",
                    index=csv_cols.index(_best_guess(
                        ["dna","dna_seq","dna_sequence","nucleotide"], csv_cols)))
            with mc2:
                map_prot = st.selectbox("Protein sequence ★", csv_cols, key="imap_prot",
                    index=csv_cols.index(_best_guess(
                        ["protein","protein_seq","protein_sequence","seq"], csv_cols)))
            with mc3:
                map_lbl = _label_select(csv_cols, "imap_lbl")
            if map_dna == map_prot:
                st.error("DNA and protein sequence must be different columns.")
                mapping_ok = False
            else:
                keep = [map_dna, map_prot]
                rmap = {map_dna: "dna_sequence", map_prot: "protein_sequence"}
                if map_lbl != "(none)":
                    keep.append(map_lbl); rmap[map_lbl] = "label"
                send_df = raw_infer_df[keep].rename(columns=rmap)

        else:
            mapping_ok = False
            st.error(f"Unknown task type: {task_type}")

        # ── Mapped preview ───────────────────────────────────────────────
        if mapping_ok and send_df is not None:
            batch_signature = (
                source_run_id,
                task_type,
                batch_file.name,
                len(batch_file.getvalue()),
                tuple(send_df.columns.tolist()),
            )
            if st.session_state.get(_BATCH_EDITED_SIG_KEY) == batch_signature:
                edited_df = st.session_state.get(_BATCH_EDITED_DF_KEY)
                if isinstance(edited_df, pd.DataFrame) and list(edited_df.columns) == list(send_df.columns):
                    send_df = edited_df.copy()
                else:
                    st.session_state.pop(_BATCH_EDITED_DF_KEY, None)
                    st.session_state.pop(_BATCH_EDITED_SIG_KEY, None)
            else:
                st.session_state.pop(_BATCH_EDITED_DF_KEY, None)
                st.session_state.pop(_BATCH_EDITED_SIG_KEY, None)

            st.divider()
            st.markdown("**Mapped Preview** (first 5 rows)")
            preview = send_df.head(5).copy()
            for col in preview.select_dtypes(include="object").columns:
                preview[col] = preview[col].astype(str).str[:50] + "…"
            st.dataframe(preview, use_container_width=True, hide_index=True)
            st.caption(
                f"★ = required · sending **{len(send_df):,} rows** · "
                f"columns: `{'`, `'.join(send_df.columns.tolist())}`"
            )

            if map_lbl == "(none)":
                st.warning(
                    "**No label column mapped.** Without ground-truth labels the following "
                    "results will **not** be available after inference: Accuracy, AUROC, AUPRC, "
                    "F1, MCC, ROC curve, Precision–Recall curve, and Confusion Matrix. "
                    "Map a label column (0 = non-interacting · 1 = interacting) above to unlock them."
                )

            batch_blocked = _render_batch_validation_errors(send_df, batch_signature)
            if st.button("Run Inference", type="primary",
                         use_container_width=True, key="infer_submit_batch",
                         disabled=batch_blocked):
                csv_bytes = send_df.to_csv(index=False).encode()
                with st.spinner("Submitting inference job..."):
                    try:
                        files = [("files", (batch_file.name, csv_bytes, "text/csv"))]
                        r = requests.post(
                            f"{BACKEND}/run_inference/{source_run_id}", files=files,
                            data={"infer_label": batch_file.name},
                        )
                        r.raise_for_status()
                        data = r.json()
                        if "error" in data:
                            st.error(data["error"])
                        else:
                            st.session_state["infer_result_run_id"] = data["run_id"]
                            st.success(f"Inference job submitted — Run ID: `{data['run_id']}`")
                            st.session_state["infer_run_id"] = data["run_id"]
                    except Exception as e:
                        st.error(f"Submission failed: {e}")
    else:
        st.info("Upload a CSV file above to configure column mapping.")

# =============================================================================
# 4. View Results button
# =============================================================================

infer_run_id = st.session_state.get("infer_run_id", "")
if not infer_run_id:
    st.stop()

st.divider()
st.info(f"Inference job submitted — Run ID: `{infer_run_id}`")
if st.button("View Results", type="primary", use_container_width=True, key="infer_goto_results"):
    st.session_state["infer_result_run_id"] = infer_run_id
    st.switch_page("inference_results.py")
