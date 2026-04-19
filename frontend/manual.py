import streamlit as st

st.title("Manual")
st.caption("Step-by-step guide for using Deep-Prot Studio.")
st.divider()

with st.expander("1 — Prepare your data", expanded=True):
    st.markdown("""
Upload a CSV file with your protein sequences and interaction labels.

| Column | Description |
|--------|-------------|
| `proteinA` | Amino acid sequence of the first protein |
| `proteinB` | Amino acid sequence of the second protein |
| `label` | Interaction label: `1` (interacting) or `0` (non-interacting) |

- Accepted characters: standard IUPAC amino acid letters.
- Sequences longer than 1,022 residues are handled via a sliding-window approach automatically.
- Maximum **3,000 pairs** per training job.
""")

with st.expander("2 — Configure embedding model"):
    st.markdown("""
Select an ESM2 model for generating protein sequence embeddings.

| Model | Embedding dim | Speed |
|-------|--------------|-------|
| ESM2 8M | 320 | Fastest |
| ESM2 35M | 480 | Default |
| ESM2 150M | 640 | Accurate |
| ESM2 650M | 1280 | Slow |

Larger models produce richer embeddings but require more GPU memory and time.
""")

with st.expander("3 — Build your model architecture"):
    st.markdown("""
Use the **Model Builder** to stack layers on top of the ESM2 embeddings.

Available layer types:

- **Linear** — fully connected layer with activation and optional batch normalisation.
- **CNN1D** — 1-D convolutional layer for local feature extraction.
- **BiLSTM** — bidirectional LSTM for sequence context.
- **GRU** — gated recurrent unit (bidirectional option available).
- **Transformer** — self-attention encoder block.
- **Residual** — skip-connection block that preserves input dimension.

Re-order layers with the ↑ / ↓ buttons. Remove layers with ✕.
The architecture preview updates live showing dimensions and approximate parameter count.
""")

with st.expander("4 — Set training parameters"):
    st.markdown("""
| Parameter | Description |
|-----------|-------------|
| **Epochs** | Number of full passes over the training set (5–100). |
| **Learning rate** | Adam optimiser step size (0.001 / 0.0005 / 0.0001). |
| **Batch size** | Samples per gradient update (32 / 64 / 128). |
| **Early stopping** | Stop if validation loss does not improve for N epochs (0 = disabled). |
| **Pairs to use** | Soft cap — subsample from uploaded data (max 3,000). |
| **Training split** | Fraction of selected pairs used for training; remainder for testing. |
""")

with st.expander("5 — Submit and monitor"):
    st.markdown("""
1. Click **Submit Training Job**.
2. Save the displayed **cancel token** — it will not be shown again.
3. Navigate to **Tools → Check Results** and enter your Run ID to monitor live progress (loss, accuracy, AUROC).
4. Once completed, download the trained model (`.pt`) and embeddings (`.pkl`).
""")

with st.expander("6 — Run inference"):
    st.markdown("""
Go to **Tools → Inference**.

- Enter the **Run ID** of a completed training job.
- **Single Pair** — paste two sequences (raw or FASTA) and click *Predict Interaction*.
- **Batch CSV** — upload a CSV with `proteinA` and `proteinB` columns.

Results include probability scores, prediction labels, and (when ground-truth labels are provided) ROC/PR curves, confusion matrix, and threshold sensitivity analysis.
""")

st.divider()
st.subheader("FAQ & Troubleshooting")

faq = {
    "My job failed — what should I do?": """
**Common causes:**

- **Out of GPU memory** — your dataset is too large for the selected ESM2 model. Try reducing *Pairs to use* or switching to a smaller model (e.g. ESM2 8M instead of ESM2 650M).
- **Invalid sequences** — sequences containing non-standard characters (e.g. `*`, `-`, `X` in excess) can cause embedding errors. Re-check your CSV.
- **Timeout** — jobs have a 4-hour hard limit. Very large datasets with a large ESM2 model may exceed this. Reduce dataset size or use a faster model.
- **Worker crash** — if the backend worker restarted mid-job, the job is automatically marked as failed. Simply resubmit.

Check the job error message in **Tools → Job Status** for a specific hint.
""",

    "What ESM2 model should I pick?": """
| Model | Dim | Best for |
|-------|-----|---------|
| ESM2 8M | 320 | Quick experiments, very large datasets (>2,000 pairs) |
| ESM2 35M | 480 | Default — good balance of speed and accuracy |
| ESM2 150M | 640 | When accuracy matters and you have <1,000 pairs |
| ESM2 650M | 1280 | Small, high-quality datasets where maximum accuracy is needed |

**Rule of thumb:** start with 35M. If training is too slow or runs out of memory, drop to 8M. If results are unsatisfactory and your dataset is small, try 150M.
""",

    "What does AUROC mean? What is a good score?": """
**AUROC** (Area Under the Receiver Operating Characteristic curve) measures how well the model separates interacting from non-interacting pairs across all possible thresholds.

| AUROC | Interpretation |
|-------|---------------|
| 0.50 | Random guess — model learned nothing |
| 0.60–0.70 | Poor |
| 0.70–0.80 | Acceptable |
| 0.80–0.90 | Good |
| 0.90–1.00 | Excellent |

AUROC is **threshold-independent** — it does not assume any particular cutoff. Use it to compare models. Use **F1** or **Precision/Recall** when your operating threshold matters.
""",

    "What is Average Precision (AP) and how is it different from AUROC?": """
**Average Precision (AP)** is the area under the Precision-Recall curve. It is more informative than AUROC when your dataset is **imbalanced** (far more negatives than positives, which is common in biology).

- A model with AUROC = 0.90 can still have AP = 0.40 if positives are very rare.
- Always check AP alongside AUROC for biological datasets.
- AP closer to 1.0 is better; a random classifier achieves AP ≈ fraction of positives in your data.
""",

    "What is MCC and why should I use it?": """
**Matthews Correlation Coefficient (MCC)** is a single balanced metric that accounts for all four cells of the confusion matrix (TP, TN, FP, FN). It is robust to class imbalance.

| MCC | Interpretation |
|-----|---------------|
| −1 | Perfect inverse prediction |
| 0 | No better than random |
| +1 | Perfect prediction |

MCC is the recommended metric when your dataset has unequal class sizes. It penalises a model that simply predicts the majority class for every sample.
""",

    "My model has high accuracy but poor AUROC — why?": """
This usually happens with **imbalanced datasets**. If 90 % of your pairs are non-interacting, a model that always predicts 0 achieves 90 % accuracy but AUROC ≈ 0.50.

**Solutions:**
- Check the class balance in the Dataset Overview section of Check Results.
- Use AUROC, AP, and MCC as primary metrics instead of accuracy.
- If your positive-to-negative ratio is worse than 1:9, consider reducing negatives or increasing positives in your CSV.
""",

    "How do I choose between pair representation modes (PPI)?": """
The **pair representation** controls how the two protein embeddings are combined before being fed into the classifier.

| Mode | Formula | Input size | Notes |
|------|---------|-----------|-------|
| `concat` | [eA, eB] | dim × 2 | Simple; direction-sensitive (A vs B order matters) |
| `product` | eA ⊙ eB | dim × 1 | Captures co-occurrence features |
| `diff` | \|eA − eB\| | dim × 1 | Captures dissimilarity |
| `all` | [concat, product, diff, sum] | dim × 4 | **Recommended** — richest representation |

Start with `all`. If training is very slow due to the larger input size, try `concat`.
""",

    "How many training pairs do I need?": """
As a rough guide:

| Pairs | Expected outcome |
|-------|-----------------|
| < 50 | Likely to overfit; results unreliable |
| 50–200 | Usable for quick experiments; validate carefully |
| 200–1,000 | Good for most tasks with a simple architecture |
| 1,000–3,000 | Recommended for robust models |

Balance matters more than absolute count. A 500-pair balanced dataset typically outperforms a 2,000-pair heavily imbalanced one.
""",

    "What is early stopping and how should I set it?": """
**Early stopping** monitors validation loss after each epoch. If it does not improve for **N consecutive epochs**, training stops — even if the maximum epoch count has not been reached.

- Set to **0** to disable and always run the full epoch budget.
- A value of **5–10** is a good default. It prevents overfitting and saves GPU time.
- If your training curves show validation loss rising while training loss falls, reduce this number (the model is already overfitting).
- If training stops too early and metrics are still improving, increase this number.
""",

    "Can I run inference without ground-truth labels?": """
Yes. Upload a CSV with only the sequence columns (no `label` column). The platform will return predicted probabilities and binary predictions (threshold = 0.5) for every pair.

When labels **are** provided, additional diagnostics are shown: ROC curve, PR curve, confusion matrix, AUROC, F1, MCC, and a threshold sensitivity slider.
""",

    "How do I compare multiple trained models?": """
Go to **Tools → Model Comparison**.

1. Paste up to 5 Run IDs into the input box and click **Add** for each.
2. Click **Load / Refresh Comparison**.

You will see a summary metrics table, overlaid ROC and PR curves, training curves, side-by-side confusion matrices, and architecture cards for each model — all on one page.
""",

    "My Run ID is lost — can I find it again?": """
Yes. Go to **Tools → Job Status**. All submitted jobs are listed there with their Run IDs, task types, submission times, and final metrics. You can copy the Run ID directly from the table.

Note: the **cancel token** is shown only once at submission. If it is lost, the job cannot be cancelled — but it will complete normally.
""",

    "What file formats are supported for download?": """
| Artefact | Format | Use |
|---------|--------|-----|
| Model weights | `.pt` (PyTorch) | Load with `torch.load()` for custom inference |
| Embeddings | `.pkl` (pickle) | Python dict mapping sequence → numpy array |
| Inference results | `.csv` | Probability scores + binary predictions per pair |
""",
}

for question, answer in faq.items():
    with st.expander(question):
        st.markdown(answer)
