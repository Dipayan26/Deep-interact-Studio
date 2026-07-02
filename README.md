# Deep-Interact Studio

![Deep-Interact Studio home screen](frontend/Homepage.png)

**Deep-Interact Studio** is a unified, web-based deep learning platform for biomolecular interaction prediction. It shifts the paradigm from fixed, single-model prediction to a user-driven, comparative, and interpretable approach — allowing researchers to design their own model architectures, train them on custom data, and compare multiple models side by side, all without writing a single line of code.

**Web Application:** https://deepinteract.compbiosysnbu.in/

---

## Supported Interaction Types

| Interaction | Abbreviation |
|---|---|
| Protein–Protein Interaction | PPI |
| Drug–Target Protein Interaction | DTPI |
| RNA–Protein Interaction | RPI |
| Protein–DNA Interaction | PDI |

---

## Key Features

- **User-customizable model builder** — construct classifier architectures layer by layer with support for Linear, CNN1D, BiLSTM, GRU, Transformer, and Residual blocks, each with per-layer control over dimensionality, activation, dropout, and normalization
- **Pre-trained encoders** — sequences are encoded using frozen ESM-2 (proteins) and ChemBERTa (small molecules/drugs); a dedicated RNA-specific encoder handles RNA-protein tasks
- **Multi-model training** — train multiple architectures in parallel under identical data partitions and hyperparameters for fair comparison
- **Multi-model inference** — run up to 5 trained models simultaneously on shared input; outputs are aligned at the sample level with per-pair disagreement scoring to flag uncertain predictions
- **Integrated interpretability** — SHAP-based feature attribution, UMAP projection of input and learned feature spaces, and an interaction hub network view that highlights high-degree hub targets
- **Reproducible model registry** — every training run is stored with its architecture, configuration, metrics, and learned embeddings
- **Export-ready outputs** — all metrics, plots, and ranked tables are exportable in machine-readable formats

---

## Platform Architecture

Deep-Interact Studio is built on a layered, service-oriented architecture:

- **Frontend:** Streamlit
- **Backend:** FastAPI (validation and job management)
- **Task queue:** Celery workers (GPU-accelerated) with Redis broker
- **Database:** PostgreSQL (job metadata and results)
- **Deployment:** Docker Compose behind nginx

The end-to-end pipeline is organized into five stages:

1. **Data Preparation** — upload, mapping, validation, quality checks, and stratified train/test splitting with automatic negative pair generation for imbalanced datasets
2. **Training Module** — embedding configuration, layer-by-layer model architecture building, hyperparameter configuration, job submission, and live training metrics
3. **Model Insights** — artifact storage, architecture summary, evaluation metrics, feature-space UMAP, and multi-model comparison
4. **Inference Engine** — model selection, pair/batch inference input, validation, job submission, and prediction output
5. **Visualization & Analysis** — score distributions, prediction evaluation (ROC/PR curves, confusion matrix, AUROC/MCC), interaction hub network, inference comparison across runs, and export/downloads

---


## Benchmark Datasets

Curated PPI, RNA-protein, and drug-target interaction datasets used for validation are freely available at:
https://deepinteract.compbiosysnbu.in/benchmark_datasets

---

## Citation

If you use Deep-Interact Studio in your research, please cite:

```bibtex
@article{sarkar2025deepinteract,
  author  = {Sarkar, Dipayan and Bardhan, Koushik and Sarkar, Chiranjib},
  title   = {Deep-Interact Studio: An Interactive Deep Learning model building Platform for Biomolecular Interaction Prediction},
  year    = {2025},
  url     = {https://deepinteract.compbiosysnbu.in/}
}
```

