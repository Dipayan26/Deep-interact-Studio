"""
Inference page — submit inference jobs against a completed training run.
Results are viewed on the Inference Results page.
"""

import io
import os

import pandas as pd
import requests
import streamlit as st

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

if task_type == "dtpi":
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
        elif lt == "bilstm":
            h = int(cfg.get("hidden_size", 128))
            nl = int(cfg.get("num_layers", 1))
            gate = 4
            dirs = 2
            total += dirs * gate * (cur * h + h * h + 2 * h)
            for _ in range(nl - 1):
                total += dirs * gate * (dirs * h * h + h * h + 2 * h)
            cur = dirs * h
        elif lt == "gru":
            h = int(cfg.get("hidden_size", 128))
            nl = int(cfg.get("num_layers", 1))
            bidir = bool(cfg.get("bidirectional", True))
            gate = 3
            dirs = 2 if bidir else 1
            total += dirs * gate * (cur * h + h * h + 2 * h)
            for _ in range(nl - 1):
                total += dirs * gate * (dirs * h * h + h * h + 2 * h)
            cur = dirs * h
        elif lt == "transformer":
            d  = int(cfg.get("d_model", 256))
            ff = int(cfg.get("dim_feedforward", d * 2))
            nl = int(cfg.get("num_layers", 2))
            total += cur * d + d + nl * (4 * d * d + 4 * d + d * ff + ff + ff * d + d + 4 * d)
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
input_mode = st.radio(
    "Input Mode",
    ["Batch CSV", "Single Pair"],
    horizontal=True
)

if input_mode == "Single Pair":
    _ik = st.session_state["infer_input_key"]

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
                            data={"is_single": "true"},
                        )
                        r.raise_for_status()
                        data = r.json()
                        if "error" in data:
                            st.error(data["error"])
                        else:
                            st.success(f"Job submitted — Run ID: `{data['run_id']}`")
                            st.session_state["infer_run_id"] = data["run_id"]
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
                            data={"is_single": "true"},
                        )
                        r.raise_for_status()
                        data = r.json()
                        if "error" in data:
                            st.error(data["error"])
                        else:
                            st.success(f"Job submitted — Run ID: `{data['run_id']}`")
                            st.session_state["infer_run_id"] = data["run_id"]
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
                            data={"is_single": "true"},
                        )
                        r.raise_for_status()
                        data = r.json()
                        if "error" in data:
                            st.error(data["error"])
                        else:
                            st.success(f"Job submitted — Run ID: `{data['run_id']}`")
                            st.session_state["infer_run_id"] = data["run_id"]
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
                            data={"is_single": "true"},
                        )
                        r.raise_for_status()
                        data = r.json()
                        if "error" in data:
                            st.error(data["error"])
                        else:
                            st.success(f"Job submitted — Run ID: `{data['run_id']}`")
                            st.session_state["infer_run_id"] = data["run_id"]
                    except Exception as e:
                        st.error(f"Submission failed: {e}")

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

            if st.button("Run Inference", type="primary",
                         use_container_width=True, key="infer_submit_batch"):
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
