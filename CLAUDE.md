# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Deep-Prot Studio** — a GPU-accelerated multi-task bioinformatics platform for training and running deep learning models on biological sequence data. Currently implements four active tasks: **PPI** (Protein-Protein Interaction), **DTPI** (Drug-Target Protein Interaction), **RPI** (RNA-Protein Interaction), and **PDI** (Protein-DNA Interaction). Other tasks (subcellular localization, protein function) remain placeholders with "Coming Soon" status.

Users upload CSVs → embeddings generated on GPU → MLP classifier trained → model/embeddings downloadable. Inference on new pairs using a trained model is also supported.

## Running the Project

```bash
# Start all services (recommended)
docker compose up --build
docker compose up --build -d   
# Rebuild without cache (use when code changes)
docker compose build --no-cache

# Rebuild a single service (faster)
docker compose build --no-cache frontend && docker compose up -d frontend
docker compose build --no-cache backend  && docker compose up -d backend

docker compose build --no-cache frontend backend celery-worker
docker compose up -d frontend backend celery-worker
# Rebuild celery worker after tasks.py changes
docker compose build --no-cache celery-worker && docker compose up -d celery-worker
```

Service URLs:
- Frontend (Streamlit): http://localhost:8502  (nginx: web3.compbiosysnbu.in)
- Backend (FastAPI):    http://localhost:8006
- PostgreSQL:           localhost:5433  (db: `ppi_jobs`, user: `ppi_user`, pass: `ppi_pass`)
- Redis:                localhost:6379

## Architecture

Five Docker services:

```
Streamlit (8502) → FastAPI (8005 internal) → Redis → Celery Worker (GPU)
                           ↕
                      PostgreSQL (5433)
```

**Key data flow (PPI):**
1. User uploads CSV + hyperparams via `frontend/ppi.py`
2. `backend/main.py` saves files, creates DB job record, enqueues Celery task, returns `{run_id, cancel_token}`
3. Celery worker (`backend/tasks.py::train_ppi_model`) runs ESM2 embedding (`backend/model_build/esm_embed.py`) → trains MLP classifier (`backend/model_build/ppi_classifier.py`)
4. Per-epoch metrics written to `/app/saved_models/{run_id}/metrics_{run_id}.json`
5. Model saved as `.pt`, embeddings as `.pkl` under `/app/saved_models/{run_id}/`
6. Frontend polls `/metrics/{run_id}` for live progress; downloads via `/download_model/` and `/download_embedding/`

**DTPI flow:** `frontend/dtpi.py` → `train_dtpi_model` task → ChemBERTa SMILES embeddings (`chemberta_embed.py`) + ESM2 protein embeddings → concat features → `FlexiblePPIModel`.

**RPI flow:** `frontend/rna_prot.py` → `train_rpi_model` task → RNA-FM embeddings + ESM2 protein embeddings → concat → `FlexiblePPIModel`.

**PDI flow:** `frontend/prot_dna.py` → `train_pdi_model` task → DNABERT embeddings + ESM2 protein embeddings → concat → `FlexiblePPIModel`.

`task_type` ∈ {`ppi`, `dtpi`, `rpi`, `pdi`} in hyperparams routes the job to the correct trainer/inference task.

## Backend API Endpoints (`backend/main.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/create_job` | Accept CSV files + hyperparams JSON, queue training task (PPI or DTI based on `task_type`), return `{run_id, cancel_token}` |
| GET  | `/check_status/{run_id}` | Poll job status from DB |
| POST | `/cancel_job/{run_id}` | Cancel job (requires cancel token, SHA-256 verified) |
| GET  | `/metrics/{run_id}` | Live per-epoch training metrics JSON (polled by frontend) |
| GET  | `/download_embedding/{run_id}` | Return `.pkl` embedding file |
| GET  | `/download_model/{run_id}` | Return `.pt` model weights |
| POST | `/run_inference/{source_run_id}` | Queue inference job; accepts `infer_label` (filename) and `is_single` (bool) form fields; stores them in inference job's `hyperparams`; routes to correct task based on source job's `task_type` |
| GET  | `/download_results/{run_id}` | Return inference results CSV |
| GET  | `/inference_metrics/{run_id}` | Predicted probabilities + (optional) ground-truth labels + AUROC/AUPRC/F1/accuracy/MCC for the inference dashboard |
| GET  | `/jobs` | List all jobs (training + inference) with final metrics, task_type, and architecture info |
| GET  | `/health` | Health check |

## Frontend Pages (`frontend/`)

`app.py` — entry point, `st.navigation()` with grouped sidebar:
- **""** (top): `home.py`, `manual.py`, `contact.py`
- **Data**: `benchmark_datasets.py`
- **Model Building**: `ppi.py`, `dtpi.py`, `rna_prot.py`, `prot_dna.py`, `subcellular.py` (coming soon), `protein_function.py` (coming soon)
- **Tools**: `job_status.py`, `check_results.py`, `inference.py`, `inference_results.py`, `comparison.py`, `inference_comparison.py`

Active pages:
- `ppi.py` — PPI training: CSV upload → column mapping → validation → data sampling (pairs slider + single positive% slider) → ESM2 model + embedding mode (pooled/chunked) → model builder → submit. Shows warning if pooled embedding is used with BiLSTM/GRU/Transformer layers.
- `dtpi.py` — DTPI training (smiles/sequence/label columns; ChemBERTa + ESM2 model selection)
- `rna_prot.py` — RPI training (rna_sequence/protein_sequence/label columns; RNA-FM + ESM2)
- `prot_dna.py` — PDI training (dna_sequence/protein_sequence/label columns; DNABERT + ESM2)
- `inference.py` — **Submission-only**: select a completed training run, upload CSV or enter a single pair, submit inference job, then navigate to results via "View Results" button. Does not display results.
- `inference_results.py` — **Results viewer**: load any completed inference run by run_id (pre-filled from job_status or inference.py). Shows full dashboard: ROC/PR curves, Confusion Matrix, Epoch Curves, KDE, SHAP, Probability Distribution, Score Scatter, Raw Results. Shows sibling inference runs from the same source model in an expander.
- `check_results.py` — Monitor a training run: status badge, progress bar, loss/accuracy charts, final metrics, download buttons, cancel form
- `job_status.py` — Table of all submitted jobs. View button routes training → `check_results.py`, inference → `inference_results.py`. Inference rows show source model run_id below the "Inference" badge.
- `comparison.py` — Side-by-side comparison of multiple completed training runs
- `inference_comparison.py` — Side-by-side comparison of multiple completed inference runs (same task type required)
- `home.py` — Platform landing page with task cards
- `benchmark_datasets.py` — Benchmark dataset browser
- `subcellular.py`, `protein_function.py` — Coming-soon placeholders

Shared helper modules:
- `data_sampling.py` — `balanced_sample_by_label`, `compute_balanced_sample_counts` (used by all task pages)
- `validation_recovery.py` — CSV validation and row-recovery UI (long sequences, invalid chars)
- `architecture_graph.py` — Model architecture visualizer
- `model_builder_defaults.py` — Default layer configs and reset helpers

## Key Implementation Details

- **ESM2 models**: Multiple sizes selectable (e.g. `esm2_t12_35M_UR50D` default, also `esm2_t6_8M_UR50D`). Loaded lazily into `/app/hf_cache` via HuggingFace; cache is volume-mounted (`./backend/hf_cache:/app/hf_cache`).
- **ChemBERTa**: `seyonec/ChemBERTa-zinc-base-v1` default — 768-dim mean-pooled output (`CHEMBERTA_DIM = 768`). Lazy load/unload mirrors `esm_embed.py`.
- **Sliding window**: Sequences > 1022 residues use a sliding window (stride 512) in `esm_embed.py`.
- **PPI pair representation**: `concat` / `product` / `diff` / `all` (recommended). `all` = ESM_dim × 4 input.
- **DTI feature vector**: `concat(chem_embedding, esm_embedding)` — fed into the same `FlexiblePPIModel` from `ppi_classifier.py`.
- **MLP classifier**: `backend/model_build/ppi_classifier.py` (`FlexiblePPIModel`) — configurable depth/width, ReLU + Dropout(0.3), class-weighted CrossEntropyLoss, per-epoch JSON metrics, early stopping on val loss. Reused by DTI.
- **Inference**: `backend/model_build/ppi_infer.py` and `dti_infer.py` — load saved model, embed any new sequences/SMILES not present in the source run's cached embedding dict, return probabilities. Both write `infer_metrics_{run_id}.json` with probabilities + per-metric scores when labels are present.
- **Celery**: Redis broker + result backend. 4-hour soft/hard time limits. `SoftTimeLimitExceeded` marks job as failed gracefully. Tasks: `train_ppi_model`, `run_ppi_inference`, `train_dtpi_model`, `run_dtpi_inference_task`, `train_rpi_model`, `run_rpi_inference_task`, `train_pdi_model`, `run_pdi_inference_task`.
- **Cancel token**: `secrets.token_urlsafe(32)` shown once to user; SHA-256 hash stored in DB. Cancel verified by re-hashing and comparing. Celery task revoked via `celery.control.revoke(terminate=True)`.
- **NaN sanitization**: `_safe(v)` helper in both `ppi_classifier.py` and `main.py` converts numpy floats and replaces NaN/Inf with `None` before JSON serialization.
- **DB migration**: `startup()` in `main.py` runs `ALTER TABLE jobs ADD COLUMN IF NOT EXISTS` for all new columns — safe to run on existing DBs.
- **Cleanup scheduler** (`_run_cleanup` + `_start_cleanup_scheduler` in `main.py`): Runs once at startup, then **every hour** via a daemon background thread — no restart needed. TTLs:
  - `running` jobs older than 5 h → marked failed, dir deleted
  - `queued` jobs older than 1 d → marked failed, dir deleted
  - `failed` / `cancelled` → dir deleted immediately; DB row purged after **1 day**
  - `completed` → dir + DB row purged after **7 days**
- **Data sampling UI**: All task pages use a single "Positive pairs (%)" slider (5–95%, step 5) with a `st.metric` showing the complementary negative %. The old two-slider linked design has been removed. When `_max_pairs ≤ 20`, the pairs slider is replaced with a fixed info message.
- **Inference label**: When submitting an inference job, the uploaded filename is sent as `infer_label` form field and stored in the inference job's `hyperparams`. `is_single=true` is sent for single-pair submissions. Both are read back by `inference_results.py` via `/job_detail`.
- **Column mapping**: Frontend accepts any CSV column names; user maps to required fields via dropdowns; CSV is renamed before sending to backend. Supported schemas: PPI (`proteinA`, `proteinB`, `label`), DTPI (`smiles`, `sequence`, `label`), RPI (`rna_sequence`, `protein_sequence`, `label`), PDI (`dna_sequence`, `protein_sequence`, `label`).
- **GPU**: Backend and Celery worker use `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime` with NVIDIA GPU access via `deploy.resources.reservations.devices`.

## Database Model (`backend/models.py`)

`Job` table columns: `run_id`, `status`, `job_type` (train/inference), `input_sequence`, `hyperparams` (includes `task_type`, `esm_model`, `chem_model`, `layer_configs`, etc.), `model_path`, `metrics`, `source_run_id`, `cancel_token_hash`, `celery_task_id`, `result`, `created_at`

## Configuration

All configuration is hardcoded or via Docker Compose environment variables — no `.env` file:
- `BACKEND_URL` env var used by frontend (default: `http://backend:8005`)
- Database credentials hardcoded in `backend/database.py`
- `HF_HOME=/app/hf_cache` (set in `chemberta_embed.py` before transformers import)
- `backend/saved_models/` and `backend/hf_cache/` are git-ignored and volume-mounted

## Nginx

Production: `web3.compbiosysnbu.in` proxied via `/etc/nginx/sites-available/web3`
- Frontend: `proxy_pass http://127.0.0.1:8502/`
- Backend API: `proxy_pass http://127.0.0.1:8006/`

## Common Gotchas

- Always use `docker compose` (v2), never `docker-compose` (v1) — v1 has bugs with newer images and GPU syntax.
- After changing `backend/tasks.py` or `backend/model_build/`, rebuild **both** `backend` and `celery-worker`.
- FastAPI's JSON encoder rejects `float('nan')` — always pass through `_safe()` before returning metrics.
- The `cancel_token` is returned only once at job creation. If frontend session is lost, user cannot cancel (by design — no auth system).
- Inference jobs store `task_type` inside their own `hyperparams` JSON (not as a top-level DB column). Always read from `hp.get("task_type")` — never fall back to `job_type` (which is always `"inference"`). This distinction matters in `inference_comparison.py` and `inference_results.py`.
- ChemBERTa hidden size is 768 (not 384) — `chem_dim` hyperparam should match the loaded model.
- `inference.py` is **submission-only**; it does not render results. Results live in `inference_results.py`. Do not add result-rendering back to `inference.py`.
- The cleanup scheduler runs as a daemon thread — it does not need a cron job or Celery Beat. It is started once in `startup()` via `_start_cleanup_scheduler()`.

# rebuilding
- you dont need to run docker compose , i will run it myself/
