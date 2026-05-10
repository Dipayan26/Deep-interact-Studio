import streamlit as st


TASKS = {
    "PPI": {
        "name": "Protein-Protein Interaction",
        "input": "`ProteinA`, `ProteinB`, `lable`",
        "meaning": "Predict whether two protein sequences interact.",
        "page": "ppi.py",
        "icon": ":material/hub:",
    },
    "DTPI": {
        "name": "Drug-Target Protein Interaction",
        "input": "`SMILES`, `Protein`, `lable`",
        "meaning": "Predict whether a compound binds a protein target.",
        "page": "dtpi.py",
        "icon": ":material/medication:",
    },
    "RPI": {
        "name": "RNA-Protein Interaction",
        "input": "`RNA`, `Protein`, `lable`",
        "meaning": "Predict whether an RNA sequence binds a protein.",
        "page": "rna_prot.py",
        "icon": ":material/genetics:",
    },
    "PDI": {
        "name": "Protein-DNA Interaction",
        "input": "`DNA`, `Protein`, `lable`",
        "meaning": "Predict whether a protein binds a DNA sequence.",
        "page": "prot_dna.py",
        "icon": ":material/biotech:",
    },
}


st.title("Manual")
st.caption("A compact guide to using Deep-Interact Studio.")

st.info(
    "Workflow: choose an interaction task, upload or load data, map columns, configure the model, "
    "submit training, check results, then run inference on new pairs."
)

st.header("What This Webtool Does")
st.markdown(
    """
Deep-Interact Studio builds sequence-based interaction classifiers for biological pair prediction.
It turns molecules and sequences into embeddings, trains a classifier, reports validation metrics,
and lets you reuse completed models for inference.

The app supports four tasks:
"""
)

cols = st.columns(4)
for col, (task_id, task) in zip(cols, TASKS.items()):
    with col:
        st.container(border=True).markdown(
            f"**{task_id}**  \n"
            f"{task['name']}  \n\n"
            f"{task['meaning']}"
        )

st.divider()

st.header("1. Prepare Data")
st.markdown(
    """
Use a CSV with two input columns and one binary label column. The label is `1` for interaction
and `0` for no interaction. Column names do not need to match exactly because each model-building
page lets you map your columns after upload.
"""
)

tab_ppi, tab_dtpi, tab_rpi, tab_pdi = st.tabs(list(TASKS.keys()))
for tab, task in zip([tab_ppi, tab_dtpi, tab_rpi, tab_pdi], TASKS.values()):
    with tab:
        st.markdown(f"**Use for:** {task['meaning']}")
        st.markdown(f"**Expected data:** {task['input']}")

st.markdown(
    """
Practical checks before training:
- Keep labels as `0` and `1`.
- Remove duplicate pairs where possible.
- Use balanced positives and negatives for the first experiment.
- Protein inputs are validated against the current page limit of 512 residues.
- Start with a small sample if the CSV is large.
"""
)

st.page_link(
    "benchmark_datasets.py",
    label="Open Benchmark Datasets",
    icon=":material/dataset:",
)

st.divider()

st.header("2. Train A Model")
st.markdown("Open the page for your task and follow the same basic sequence.")

step_cols = st.columns(4)
for col, (task_id, task) in zip(step_cols, TASKS.items()):
    with col:
        st.page_link(task["page"], label=f"Open {task_id}", icon=task["icon"])

st.markdown(
    """
On a model-building page:
1. Upload a CSV or load the example data.
2. Map the sequence, molecule, and label columns.
3. Choose how many pairs to use and the train/test split.
4. Select the embedding model where the page offers options.
5. Build a classifier from the available layers.
6. Set training parameters such as epochs, batch size, and learning rate.
7. Submit the job and save the Run ID.
"""
)

with st.expander("Recommended starting settings"):
    st.markdown(
        """
| Setting | Good first choice |
|---|---|
| Dataset sample | 100-500 pairs for a first test |
| Train/test split | 80/20 |
| Class balance | 50% positive, 50% negative |
| Protein embedding | ESM2 35M when available |
| Classifier | Two Linear layers with dropout |
| Epochs | 30-50 with early stopping |
| Learning rate | 0.001 |
"""
    )

st.divider()

st.header("3. Track Results")
st.markdown(
    """
After submission, use the Run ID to follow the job and inspect model quality.

- **Job Status** lists submitted training and inference jobs.
- **Check Model Results** shows training curves, metrics, dataset summary, downloads, and failures.
- Main metrics to compare are AUROC, Average Precision, MCC, F1, and the confusion matrix.
"""
)

status_col, results_col = st.columns(2)
with status_col:
    st.page_link("job_status.py", label="Open Job Status", icon=":material/list_alt:")
with results_col:
    st.page_link(
        "check_results.py",
        label="Open Check Model Results",
        icon=":material/monitor_heart:",
    )

with st.expander("How to read the main metrics"):
    st.markdown(
        """
| Metric | What it tells you |
|---|---|
| AUROC | How well positives rank above negatives across thresholds |
| Average Precision | Better than AUROC when positives are rare |
| MCC | Balanced single-score metric for imbalanced data |
| F1 | Balance between precision and recall at one threshold |
| Accuracy | Easy to understand, but misleading when classes are imbalanced |
"""
    )

st.divider()

st.header("4. Run Inference")
st.markdown(
    """
Use a completed training Run ID to predict new pairs. You can enter a single pair manually or
upload a batch CSV. If labels are included in a batch file, the app also reports inference metrics.
"""
)

infer_col, infer_results_col = st.columns(2)
with infer_col:
    st.page_link("inference.py", label="Open Run Inference", icon=":material/play_arrow:")
with infer_results_col:
    st.page_link(
        "inference_results.py",
        label="Open Inference Results",
        icon=":material/analytics:",
    )

st.divider()

st.header("5. Compare Runs")
st.markdown(
    """
Use comparison pages when you train multiple models or run multiple inference batches.

- **Multi-Model Comparison** compares completed training runs from the same task type.
- **Multi-Model Inference** compares completed inference runs from the same task type.
"""
)

compare_col, infer_compare_col = st.columns(2)
with compare_col:
    st.page_link(
        "comparison.py",
        label="Open Multi-Model Comparison",
        icon=":material/compare:",
    )
with infer_compare_col:
    st.page_link(
        "inference_comparison.py",
        label="Open Multi-Model Inference",
        icon=":material/difference:",
    )

st.divider()

st.header("FAQ & Troubleshooting")

with st.expander("What are the job submission limits?", expanded=True):
    st.markdown(
        """
Training jobs are accepted only when they stay within these limits:

| Limit | Current value |
|---|---|
| Upload size per file | 100 MB |
| Total upload request size | 100 MB |
| Selected training pairs | Up to 100,000 pairs |
| Model size | Up to 5,000,000 trainable parameters |
| Protein sequence length on builder pages | 512 residues |
| Training wall-clock time | 4 hours, then the job is stopped |
| Training submissions | 10 training jobs per IP per 3 hours |
| Total active training queue | 20 queued or running training jobs platform-wide |

If your job is rejected, reduce the selected positive/negative pair counts, simplify the
architecture, trim long protein sequences, or wait for queued/running jobs to finish.
"""
    )

with st.expander("What does the training queue limit mean?"):
    st.markdown(
        """
The queue limit is platform-wide. It counts all training jobs with status **queued** or
**running**, not only your own jobs. When the active training queue reaches 20 jobs, new training
submissions are temporarily blocked until at least one job completes, fails, or is cleaned up.

This protects a free shared research deployment from building a very long backlog. Your per-IP
quota still applies separately: one IP can submit up to 10 training jobs in a rolling 3-hour
window, provided the platform queue is not already full.
"""
    )

with st.expander("What are the inference limits?"):
    st.markdown(
        """
Inference jobs have separate limits from training:

| Limit | Current value |
|---|---|
| Single-pair input fields | 512 characters each |
| Batch inference CSV | Up to 60,000 pairs |
| Inference submissions | 20 inference requests per minute |
| Batch inference quota | 15 batch jobs per IP per 5 hours |
| Single-pair inference quota | 30 single-pair jobs per IP per 5 hours |

For large inference files, keep only the required columns and remove duplicate rows before upload.
"""
    )

with st.expander("How many pairs should I use?"):
    st.markdown(
        """
For a first run, use 100-500 balanced pairs to check that the workflow, column mapping, and model
settings are correct. For stronger models, increase the pair count while watching runtime and
class balance.

Very small datasets can overfit. Very large datasets can time out or exceed memory limits,
especially with larger embedding models or large classifier architectures.
"""
    )

with st.expander("What should I save after submitting a job?"):
    st.markdown(
        """
Save the **Run ID** immediately. It is needed to check status, view results, run inference, and
compare models. If a cancel token is shown after submission, save that too because it cannot be
recovered later.

If you lose a Run ID, open **Job Status** and look for the job by task type, submission time, or
final metrics.
"""
    )

with st.expander("A job failed"):
    st.markdown(
        """
Check **Job Status** or **Check Model Results** for the error. Common causes are invalid
sequences, invalid SMILES strings, too many selected pairs, GPU memory limits, or timeout.
Reduce the sample size and retry with a smaller embedding model if the job is too heavy.
"""
    )

with st.expander("Which metrics should I report?"):
    st.markdown(
        """
Report AUROC, Average Precision, MCC, and the confusion matrix. Accuracy can be included, but it
should not be the primary metric when classes are imbalanced.

For candidate screening, also inspect precision and recall at the threshold you plan to use.
Lower thresholds find more possible interactors; higher thresholds return fewer but more confident
candidates.
"""
    )

with st.expander("Results look too good"):
    st.markdown(
        """
Check for duplicate pairs and shared entities between train and test data. High overlap can make
validation metrics look better than real-world performance. For publication-style evaluation,
prefer disjoint splits where possible.
"""
    )

with st.expander("Accuracy is high but AUROC or MCC is poor"):
    st.markdown(
        """
This usually means the dataset is imbalanced. Use AUROC, Average Precision, MCC, and the
confusion matrix instead of relying only on accuracy.
"""
    )

with st.expander("How many runs can I compare?"):
    st.markdown(
        """
The comparison pages accept up to **5 runs** at a time. Compare only runs from the same task type
so the metrics and prediction outputs are meaningful.
"""
    )
