import streamlit as st

st.title("Manual")
st.caption("Step-by-step guide for using Deep-Prot Studio.")
st.divider()

st.info(
    "**Quick View: Follow this order:** "
    "Pick your task type → Prepare a CSV → Upload & map columns → "
    "Choose an embedding model → Build your classifier → Train → Check results → Run inference."
)

st.header(" What Can Deep-Prot Studio Do?")
st.markdown("""
Deep-Prot Studio lets you train **sequence-based interaction classifiers** without writing any code.
You supply labelled pairs of biological sequences; the platform converts them to numerical embeddings
using pre-trained foundation models, trains the customized model, and returns performance metrics
plus a downloadable model you can re-use for inference on new pairs.

Four task types are supported:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
🔵 Protein–Protein Interaction (PPI)
Predict whether two protein sequences physically interact.  
*Embedding:* ESM2 (both proteins)

🟦 Drug–Target Protein Interaction (DTPI)
Predict binding between a small molecule (SMILES) and a target protein.  
*Embedding:* ChemBERTa (compound) + ESM2 (protein)
""")
with col2:
    st.markdown("""
🟣 RNA–Protein Interaction (RPI)
Predict binding between an RNA sequence and an RNA-binding protein.  
*Embedding:* RNA-FM (RNA) + ESM2 (protein)

🟠Protein–DNA Interaction (PDI)
Predict binding between a DNA-binding protein and a DNA sequence.  
*Embedding:* DNABERT 6-mer (DNA) + ESM2 (protein)
""")

st.markdown("""
> **How it works under the hood :**  
> Each sequence is fed through a pre-trained "language model for molecules" that converts it into a
> fixed-length vector of numbers (an *embedding*). The two embeddings are concatenated and passed
> through a the  (your *classifier*) that learns to output 1 (interacting) or 0
> (non-interacting).
""")
st.divider()
# ════════════════════════════════════════════════════════════════════════════
st.header("1.  Preparing Your Data")

with st.expander("1a — Required CSV format (all task types)", expanded=True):
    st.markdown("""
Every task expects a **CSV file** with three columns. Column names can be anything — you map them
to the correct role in the interface. Below are the requirements for each task type.
""")
    tab1, tab2, tab3, tab4 = st.tabs([
    "🔵 PPI", 
    "🟦 Drug–Target (DTPI)", 
    "🟣 RNA–Protein (RPI)", 
    "🟠 Protein–DNA (PDI)"
])

    with tab1:
        st.markdown("""
| Column role | Accepted content | Example |
|---|---|---|
| **Protein A** | Standard amino acid letters (IUPAC 20-letter alphabet) | `MKTAYIAKQRQISFVKSHFSRQ…` |
| **Protein B** | Same as above | `MSEQNNTEMTFQIQRIYTKEIS…` |
| **Label** | `1` = interacting · `0` = non-interacting | `1` |

- Sequences longer than **1,022 residues** are handled automatically via a sliding-window approach.
- Avoid non-standard characters: `*`, `-`, excessive `X` will cause embedding errors.
- Maximum **3,000 pairs** per job.
- Aim for a **balanced dataset** (roughly equal positives and negatives).
""")

    with tab2:
        st.markdown("""
| Column role | Accepted content | Example |
|---|---|---|
| **SMILES** | Valid canonical SMILES string | `CC(=O)Oc1ccccc1C(=O)O` |
| **Protein Sequence** | Standard amino acid letters | `MKTAYIAKQRQISFVKSHFSRQ…` |
| **Label** | `1` = binding · `0` = non-binding | `1` |

- Invalid SMILES strings will be flagged during validation — check them with RDKit or ChemDraw first.
- Maximum **3,000 pairs** per job.
""")

    with tab3:
        st.markdown("""
| Column role | Accepted content | Example |
|---|---|---|
| **RNA Sequence** | RNA nucleotides (A, U, C, G) or DNA (A, T, C, G — auto-converted to RNA) | `AUGCUAGCUAGCUA…` |
| **Protein Sequence** | Standard amino acid letters | `MSEQNNTEMTFQIQRIY…` |
| **Label** | `1` = binding · `0` = non-binding | `1` |

- DNA sequences (containing T) are **automatically converted** to RNA (T → U).
- Maximum **3,000 pairs** per job.
""")

    with tab4:
        st.markdown("""
| Column role | Accepted content | Example |
|---|---|---|
| **DNA Sequence** | Standard nucleotides (A, T, C, G) | `ATGCTAGCTAGCTA…` |
| **Protein Sequence** | Standard amino acid letters | `MKTAYIAKQRQISFVK…` |
| **Label** | `1` = binding · `0` = non-binding | `1` |

- Ambiguous bases (N, R, Y, …) may reduce embedding quality — use clean sequences where possible.
- Maximum **3,000 pairs** per job.
""")

with st.expander("1b — How many pairs do you need?"):
    st.markdown("""
| Dataset size | What to expect |
|---|---|
| **< 50 pairs** | Very likely to overfit; treat results as exploratory only |
| **50–200 pairs** | Useful for quick feasibility experiments; validate carefully |
| **200–1,000 pairs** | Good starting point for most biological questions |
| **1,000–3,000 pairs** | Recommended for robust, publishable models |

**Balance matters more than total count.** A 500-pair balanced dataset (250 positive + 250 negative)
will usually outperform a 2,000-pair dataset where 95 % of labels are negative.

*Where to source negative pairs if you only have positives?*  
Common strategies: random pairing of proteins not known to interact (random negatives),
phylogenetic negative sampling, or co-expression-based exclusion. The quality of your negatives
strongly influences model performance.
""")

with st.expander("1c — Data leakage and splitting strategies"):
    st.markdown("""
**Data leakage** occurs when information from the test set "leaks" into training, causing overly
optimistic metrics that do not reflect real-world performance. This is a common pitfall in
computational biology.

| Split strategy | How it works | When to use |
|---|---|---|
| **Random split** (default) | Pairs are randomly assigned to train/test | Baseline; quick experiments |
| **Protein-disjoint split** | No protein appears in both train and test sets | Recommended for generalisation; tests whether the model learns transferable features |
| **Scaffold-disjoint split** (DTPI) | No chemical scaffold appears in both splits | Required for realistic drug-discovery benchmarking |

> The platform reports a **Jaccard overlap warning** if train and validation share >30 % of entities.
> Take this warning seriously — high overlap inflates AUROC by 10–20 % in typical datasets.

**Recommendation:** always run at least one protein-disjoint evaluation before reporting results.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════════

st.header("2. Data Sampling Controls")

with st.expander("Understanding the sampling sliders", expanded=True):
    st.markdown("""
After uploading and mapping your CSV, four sliders let you control which subset is used for
training and testing.

| Slider | What it controls |
|---|---|
| **Pairs to use** | Total number of pairs sampled from your CSV (useful for quick test runs on large datasets) |
| **Training split (%)** | Fraction of selected pairs used for training; the remainder becomes the test set |
| **Positive pairs (%)** | Fraction of selected pairs that are positive (label = 1) — useful for intentional class imbalance experiments |
| **Negative pairs (%)** | Fraction of selected pairs that are negative (label = 0) |

**Practical tips:**
- Keep **Positive = Negative = 50 %** unless you are deliberately studying imbalanced scenarios.
- Use **Pairs to use** to do a fast 100-pair sanity check before committing to a full 3,000-pair run.
- A **Training split of 80 %** (80 % train, 20 % test) is a sensible default for most dataset sizes.
  Increase to 90 % only if you have fewer than 100 pairs total.
- The summary line below the sliders (`Selected N pairs → training M · testing K`) confirms your
  exact split before submission.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════════

st.header("3. Embedding Models")

with st.expander("3a — ESM2 protein embeddings (used in all four task types)", expanded=True):
    st.markdown("""
**ESM2** (Evolutionary Scale Modelling) is a protein language model trained on hundreds of millions
of protein sequences. It converts an amino acid sequence into a dense numerical vector that
captures structural and functional information without requiring 3D coordinates.

| Model | Embedding dim | Speed | Best for |
|---|---|---|---|
| **ESM2 8M** | 320 | Fastest | Quick experiments; datasets > 2,000 pairs |
| **ESM2 35M** | 480 | Default | Good balance of speed and accuracy |
| **ESM2 150M** | 640 | Accurate | Datasets < 1,000 pairs where accuracy matters |
| **ESM2 650M** | 1,280 | Slow | Small, high-quality datasets needing maximum accuracy |

**Rule of thumb:** start with **35M**. If you run out of GPU memory or training is too slow, drop
to 8M. If performance is unsatisfactory and your dataset is small, try 150M.

> Sequences longer than 1,022 residues are automatically handled via a sliding-window approach
> (the window embeddings are mean-pooled). Be aware that very long sequences (> 2,000 aa) may
> lose fine-grained local information under this scheme.
""")

with st.expander("3b — ChemBERTa compound embeddings (DTPI only)"):
    st.markdown("""
**ChemBERTa** is a BERT-style model pre-trained on SMILES strings from the ZINC database. It
converts a small molecule's SMILES representation into a 768-dimensional vector.

- Model used: `seyonec/ChemBERTa-zinc-base-v1`
- Output dimension: **768**

The compound embedding and protein (ESM2) embedding are **concatenated** to produce a
1,248-dimensional input to the classifier.

> Ensure your SMILES strings are valid canonical SMILES. Tools like RDKit (`Chem.MolToSmiles`) or
> OpenBabel can canonicalise and validate your compounds before upload.
""")

with st.expander("3c — RNA-FM RNA embeddings (RPI only)"):
    st.markdown("""
**RNA-FM** is a foundation model pre-trained on non-coding RNA sequences. It produces a
640-dimensional embedding per RNA sequence.

- Model used: `multimolecule/rnafm`
- Output dimension: **640**

The RNA embedding and protein (ESM2) embedding are **concatenated** to produce a
1,120-dimensional input to the classifier.

> DNA sequences (containing T) are automatically converted to RNA (T → U) before embedding.
> Sequences with modified bases or non-standard nucleotides may embed poorly.
""")

with st.expander("3d — DNABERT DNA embeddings (PDI only)"):
    st.markdown("""
**DNABERT** uses a 6-mer tokenisation strategy (each token = 6 consecutive nucleotides with a
sliding window). It is pre-trained on the human genome and produces a 768-dimensional embedding.

- Model used: `armheb/DNA_bert_6`
- Output dimension: **768**

The DNA embedding and protein (ESM2) embedding are **concatenated** to produce a
1,248-dimensional input to the classifier.

> DNABERT 6-mer works best on sequences that resemble genomic DNA (promoters, binding motifs,
> etc.). Very short sequences (< 20 bp) may not embed reliably.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════════

st.header("4. Model Builder")

with st.expander("4a — Available layer types", expanded=True):
    st.markdown("""
The **Model Builder** lets you stack layers on top of the concatenated embeddings to form your
classifier. Each layer takes the output of the previous layer as its input.

| Layer type | What it does | When to use |
|---|---|---|
| **Linear** | Fully connected layer with activation function and optional dropout | Good default; works well for most tasks |
| **CNN1D** | 1-D convolution — detects local motifs in the input | Useful if you suspect short-range positional features matter |
| **BiLSTM** | Bidirectional LSTM — captures sequential context in both directions | Recurrent patterns; longer-range dependencies |
| **GRU** | Gated Recurrent Unit — similar to LSTM but computationally lighter | When BiLSTM is too slow |
| **Transformer** | Self-attention encoder — models all-vs-all pairwise feature relationships | Large datasets (> 1,000 pairs); rich input representations |
| **Residual** | Skip-connection block — preserves gradients and input information | Deep architectures (≥ 4 layers); prevents vanishing gradients |

Layers can be reordered with ↑/↓ and removed with ✕. The **Architecture Preview** updates live,
showing dimensions and approximate parameter count.
""")

with st.expander("4b — Key layer parameters"):
    st.markdown("""
**hidden_dim** — the width of the layer (number of neurons). Larger = more expressive but slower
and more prone to overfitting on small datasets.

**Activation function** — the non-linearity applied after the linear transformation:

| Activation | Behaviour | Best for |
|---|---|---|
| `relu` | Sets negative values to zero | Default; robust and fast |
| `gelu` | Smooth version of ReLU | Transformer-style layers |
| `tanh` | Squashes output to [−1, 1] | Recurrent layers (LSTM, GRU) |
| `sigmoid` | Squashes output to [0, 1] | Output layer only (binary classification) |

**Dropout** — randomly zeros a fraction of neurons during training to prevent overfitting.

| Dropout value | Effect |
|---|---|
| 0.0 | No regularisation — risk of overfitting on small datasets |
| 0.2–0.3 | Mild regularisation — good for datasets > 500 pairs |
| 0.4–0.5 | Strong regularisation — recommended for datasets < 200 pairs |
| > 0.6 | Very aggressive — use only with very small data |
""")

with st.expander("4c — Recommended starter architectures"):
    st.markdown("""
These are sensible starting points. Tune from here based on your validation metrics.

**Small dataset (< 200 pairs):**
```
Linear(256, relu, dropout=0.4)
Linear(64, relu, dropout=0.4)
→ Output (sigmoid)
```
*~262 K parameters — low risk of overfitting*

**Medium dataset (200–1,000 pairs):**
```
Linear(512, relu, dropout=0.3)
Linear(128, relu, dropout=0.3)
→ Output (sigmoid)
```

**Larger dataset (> 1,000 pairs):**
```
Linear(1024, relu, dropout=0.2)
Linear(256, relu, dropout=0.2)
Linear(64, relu, dropout=0.2)
→ Output (sigmoid)
```

> If validation AUROC is much lower than training AUROC → your model is **overfitting** → increase
> dropout or reduce hidden_dim.  
> If both are low → your model is **underfitting** → increase hidden_dim or add a layer.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════════

st.header("5. Training Parameters")

with st.expander("Understanding training parameters", expanded=True):
    st.markdown("""
| Parameter | What it controls | Recommended default |
|---|---|---|
| **Epochs** | Number of complete passes through the training set | Start with 50; let early stopping decide |
| **Learning rate** | Size of each gradient-descent step | 0.001 (default); try 0.0005 if loss is unstable |
| **Batch size** | Number of pairs processed per gradient update | 64 for most datasets; 32 if GPU memory is limited |
| **Early stopping patience** | Training stops if validation loss does not improve for N consecutive epochs | 5–10 epochs |

**Learning rate intuition:**
- Too high → loss oscillates and never converges.
- Too low → training converges very slowly and may get stuck.
- Start at 0.001. If the training loss curve is noisy/unstable, halve it to 0.0005.

**Early stopping** is your friend — it prevents overfitting and saves GPU time automatically.
Set patience to 0 only if you want to study the full training curve.

> All jobs are **auto-stopped after 4 hours**. For very large datasets with ESM2 650M, consider
> using a smaller embedding model or reducing your dataset to stay within budget.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════════

st.header("6. Submit and Monitor")

with st.expander("Step-by-step submission guide", expanded=True):
    st.markdown("""
1. Review the **architecture preview** and confirm the parameter count looks reasonable for your
   dataset size (see Section 4c for guidance).
2. Optionally enter your email address to be notified when the job finishes.
3. Click **Submit Training Job**.
4. ⚠️ **Save your cancel token immediately** — it is shown only once and cannot be recovered.
   Store it alongside your Run ID in your lab notebook or a text file.
5. Note your **Run ID** (e.g. `0ba60850`).
6. Go to **Tools → Check Results** and enter the Run ID to monitor:
   - Live training and validation loss curves
   - Accuracy and AUROC per epoch
   - Dataset overview (class balance, sequence statistics)
7. Once training completes, download your:
   - **Model weights** (`.pt`) — PyTorch checkpoint for custom inference
   - **Embeddings** (`.pkl`) — pre-computed embeddings as a Python dict
   - **Results CSV** — inference predictions with probability scores

> If you lose your Run ID, go to **Tools → Job Status** — all submitted jobs are listed there with
> timestamps and metrics. Note: the cancel token cannot be recovered from Job Status.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════════

st.header("7. Run Inference")

with st.expander("Inference on new sequences", expanded=True):
    st.markdown("""
Go to **Tools → Inference** and enter the **Run ID** of a completed training job.

**Two modes are available:**

**Single Pair** — paste one pair of sequences (raw or FASTA format) and click *Predict Interaction*.
Returns: probability score + binary prediction (threshold = 0.5 by default).

**Batch CSV** — upload a CSV with the same sequence columns used during training (no label column
required). Returns a downloadable CSV with:
- `probability` — model confidence that the pair interacts (0–1)
- `prediction` — binary call at the 0.5 threshold (adjustable)

**When ground-truth labels are provided in batch mode**, additional diagnostics are shown:
- ROC curve and AUROC
- Precision-Recall curve and Average Precision
- Confusion matrix
- Threshold sensitivity slider (explore precision/recall trade-offs)
- MCC at the selected threshold

> **Threshold selection:** 0.5 is a neutral default. In biology, you often want to minimise false
> positives (validate fewer candidates in the wet lab) — in that case, raise the threshold to 0.7
> or higher to improve precision at the cost of recall. Use the threshold sensitivity slider to
> find the sweet spot for your experimental budget.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════════

st.header("8. Interpreting Your Results")

with st.expander("8a — AUROC: the primary ranking metric", expanded=True):
    st.markdown("""
**AUROC** (Area Under the Receiver Operating Characteristic curve) measures how well the model
separates interacting from non-interacting pairs **across all possible decision thresholds**.
It is threshold-independent — you do not need to choose a cutoff to compute it.

| AUROC | Interpretation |
|---|---|
| 0.50 | Random guess — the model learned nothing |
| 0.60–0.70 | Poor |
| 0.70–0.80 | Acceptable |
| 0.80–0.90 | Good |
| 0.90–1.00 | Excellent |

> AUROC is the best single metric for comparing two models on the same dataset. However, it can
> be misleadingly high on imbalanced datasets — always check AP and MCC alongside it.
""")

with st.expander("8b — Average Precision (AP): critical for imbalanced data"):
    st.markdown("""
**Average Precision (AP)** is the area under the Precision-Recall (PR) curve. It is more
informative than AUROC when your dataset is **class-imbalanced** — which is common in biology,
where experimentally confirmed interactions are sparse relative to all possible pairs.

- A model with AUROC = 0.90 can have AP = 0.40 if positives are very rare (e.g. 5 % of pairs).
- For a random classifier, AP ≈ fraction of positives in your dataset. Your model must exceed this
  baseline meaningfully to be useful.
- AP closer to 1.0 is better.

**Recommendation:** always report AP alongside AUROC in any publication or preprint.
""")

with st.expander("8c — MCC: the balanced single-number metric"):
    st.markdown("""
**Matthews Correlation Coefficient (MCC)** accounts for all four cells of the confusion matrix
(true positives, true negatives, false positives, false negatives). It is robust to class imbalance.

| MCC | Interpretation |
|---|---|
| −1 | Perfect inverse prediction |
| 0 | No better than random |
| +1 | Perfect prediction |

MCC is strongly recommended when your dataset has unequal class sizes. A model that always
predicts the majority class will have MCC ≈ 0, even if its accuracy is high.
""")

with st.expander("8d — Accuracy pitfalls"):
    st.markdown("""
**Accuracy** = (TP + TN) / total pairs.

This metric is **misleading on imbalanced datasets**. If 90 % of your pairs are non-interacting,
a model that always predicts 0 achieves 90 % accuracy but AUROC = 0.50 and MCC = 0.

> **Never use accuracy as your primary metric for biological interaction prediction.**
> Use AUROC, AP, and MCC instead.
""")

with st.expander("8e — Reading the training curves"):
    st.markdown("""
The training dashboard shows **loss and AUROC vs. epoch** for both the training set (solid line)
and validation set (dashed line).

**What to look for:**

| Pattern | Diagnosis | Action |
|---|---|---|
| Both curves improve and converge | Healthy training | Increase epochs if both are still improving |
| Validation loss rises while training loss falls | Overfitting | Increase dropout; reduce hidden_dim; add more data |
| Both curves plateau early at poor performance | Underfitting | Increase hidden_dim; add a layer; increase epochs |
| Loss oscillates wildly | Learning rate too high | Reduce learning rate to 0.0005 or 0.0001 |
| Training stops early (early stopping triggered) |  Normal — model converged | Check if validation AUROC is satisfactory |
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════════

st.header("9. Model Comparison")

with st.expander("Comparing multiple runs", expanded=True):
    st.markdown("""
Go to **Tools → Model Comparison**.

1. Paste up to **5 Run IDs** into the input box and click **Add** for each.
2. Click **Load / Refresh Comparison**.

The comparison view shows:
- Summary metrics table (AUROC, AP, MCC, accuracy, F1) for all models side-by-side
- Overlaid ROC and PR curves
- Training curves per model
- Side-by-side confusion matrices
- Architecture cards (embedding model, layer stack, parameter count)

**When to use model comparison:**
- After a hyperparameter search (different learning rates, hidden dims, dropout values).
- To compare embedding model sizes (ESM2 35M vs 150M).
- To compare task representations (PPI using two different pair-fusion strategies).
- Before finalising the model for inference on new data.

> Save your Run IDs in a spreadsheet alongside the key hyperparameters — treating each run as an
> experiment entry in your lab notebook makes reproducibility much easier.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════════

st.header("10. Downloadable Files")

with st.expander("File formats reference", expanded=True):
    st.markdown("""
| Artefact | Format | How to use |
|---|---|---|
| **Model weights** | `.pt` (PyTorch) | `model = torch.load('model.pt'); model.eval()` |
| **Embeddings** | `.pkl` (Python pickle) | `import pickle; emb = pickle.load(open('embeddings.pkl','rb'))` — returns a dict mapping sequence string → numpy array |
| **Inference results** | `.csv` | Open in Excel, R, or pandas; contains `proteinA`, `proteinB`, `probability`, `prediction` columns |

> Embeddings are cached per sequence — if you re-run inference on sequences already embedded
> during training, they are retrieved from cache, making inference faster.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════════

st.header(" FAQ & Troubleshooting")

faq = {
    "My job failed — what should I do?": """
Go to **Tools → Job Status** and look at the error message for your Run ID.

**Common causes:**

| Error hint | Likely cause | Fix |
|---|---|---|
| Out of GPU memory / OOM | Dataset too large for selected ESM2 model | Reduce *Pairs to use* or switch to ESM2 8M |
| Invalid sequences / embedding error | Non-standard characters (`*`, `-`, excessive `X`) | Clean your CSV; use standard IUPAC letters only |
| Timeout (4-hour limit) | Very large dataset + large model | Reduce dataset size or use a faster embedding model |
| Worker crash | Backend restarted mid-job | Resubmit — this is rare and automatic |
| Invalid SMILES (DTPI) | Malformed compound strings | Validate with RDKit before upload |
""",

    "What ESM2 model should I pick?": """
| Model | Dim | Best for |
|---|---|---|
| ESM2 8M | 320 | Quick experiments; very large datasets (> 2,000 pairs) |
| ESM2 35M | 480 | Default — good balance of speed and accuracy |
| ESM2 150M | 640 | When accuracy matters and dataset < 1,000 pairs |
| ESM2 650M | 1,280 | Small, high-quality datasets where maximum accuracy is needed |

Start with 35M. Drop to 8M if training is too slow or memory runs out. Try 150M if results are
unsatisfactory and your dataset is small.
""",

    "What is the Jaccard overlap warning?": """
The **Jaccard overlap** measures how many protein (or compound/RNA/DNA) entities appear in
**both** the training and validation sets.

- A high Jaccard (e.g. > 30–50 %) means the model is being evaluated on sequences it has
  essentially "seen" during training — performance will appear better than it truly is.
- For publication, aim for **protein-disjoint** splits (Jaccard = 0) to test true generalisation.
- For exploratory work, random splits with high overlap are acceptable as long as you do not
  over-interpret the metrics.
""",

    "My model has high accuracy but poor AUROC — why?": """
This is almost always a **class imbalance** problem. If 90 % of your pairs are non-interacting,
a model predicting 0 for everything achieves 90 % accuracy but AUROC ≈ 0.50.

**Solutions:**
- Check the class balance in the Dataset Overview in Check Results.
- Use AUROC, AP, and MCC as primary metrics instead of accuracy.
- If positive-to-negative ratio is worse than 1:9, consider balancing your dataset before upload
  using the *Positive/Negative pairs (%)* sliders.
""",

    "What threshold should I use for binary predictions?": """
The default threshold is **0.5** — any pair with a predicted probability ≥ 0.5 is called
positive.

Adjust based on your experimental priorities:

| Goal | Recommended threshold |
|---|---|
| Maximise recall (find as many true interactors as possible, accept more false positives) | 0.3–0.4 |
| Balanced (default) | 0.5 |
| Maximise precision (only confident predictions, fewer candidates for wet-lab follow-up) | 0.65–0.8 |

Use the **threshold sensitivity slider** in the inference results view to see exactly how
precision, recall, F1, and MCC change as you move the threshold.
""",

    "Can I run inference without ground-truth labels?": """
Yes. Upload a CSV with only the sequence columns — no label column needed. The platform returns
predicted probabilities and binary predictions for every pair.

When labels **are** provided, additional diagnostics are shown: ROC and PR curves, confusion
matrix, AUROC, AP, F1, MCC, and a threshold sensitivity slider.
""",

    "How do I compare multiple trained models?": """
Go to **Tools → Model Comparison**. Paste up to 5 Run IDs and click **Add** for each, then
**Load / Refresh Comparison**.

You will see a summary metrics table, overlaid ROC and PR curves, training curves, side-by-side
confusion matrices, and architecture cards for each model.
""",

    "My Run ID is lost — can I find it?": """
Yes. Go to **Tools → Job Status** — all submitted jobs are listed with Run IDs, task types,
submission timestamps, and final metrics.

Note: the **cancel token** is shown only once. If lost, you cannot cancel the job — but it will
complete normally and be available in Job Status when done.
""",

    "What is a good architecture for my task?": """
Start simple. A two-layer linear classifier with dropout is surprisingly effective for most
biological interaction tasks, especially with small-to-medium datasets. Complex architectures
(BiLSTM, Transformer) require more data to outperform simple linear models.

Rule of thumb: if you have < 500 pairs, use two Linear layers with dropout 0.3–0.5.
Only add recurrent or attention layers if you have > 1,000 pairs and linear models plateau.
""",

    "How should I cite Deep-Prot Studio in my paper?": """
Check the **About** page (or the platform homepage) for the recommended citation. If no
published paper is listed yet, cite the platform URL and access date as a software resource
in your methods section. Also report: task type, embedding model used, classifier architecture,
dataset size, train/test split strategy, and key metrics (AUROC, AP, MCC).
""",
}

for question, answer in faq.items():
    with st.expander(question):
        st.markdown(answer)
