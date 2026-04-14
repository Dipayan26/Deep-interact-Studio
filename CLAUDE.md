# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Deep-Prot Studio** — a GPU-accelerated multi-task bioinformatics platform for training and running deep learning models on biological sequence data. Currently implements PPI (Protein-Protein Interaction) prediction; other tasks are placeholders with "Coming Soon" status.

Users upload protein sequence CSVs → ESM2 embeddings generated on GPU → MLP classifier trained → model/embeddings downloadable. Inference on new pairs using a trained model is also supported.

## Running the Project

```bash
# Start all services (recommended)
docker compose up --build

# Rebuild without cache (use when code changes)
docker compose build --no-cache

# Rebuild a single service (faster)
docker compose build --no-cache frontend && docker compose up -d frontend
docker compose build --no-cache backend  && docker compose up -d backend

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

**Key data flow:**
1. User uploads CSV + hyperparams via `frontend/build.py`
2. `backend/main.py` saves files, creates DB job record, enqueues Celery task, returns `{run_id, cancel_token}`
3. Celery worker (`backend/tasks.py`) runs ESM2 embedding (`backend/model_build/esm_embed.py`) → trains MLP classifier (`backend/model_build/ppi_classifier.py`)
4. Per-epoch metrics written to `/app/saved_models/{run_id}/metrics_{run_id}.json`
5. Model saved as `.pt`, embeddings as `.pkl` under `/app/saved_models/{run_id}/`
6. Frontend polls `/metrics/{run_id}` for live progress; downloads via `/download_model/` and `/download_embedding/`

## Backend API Endpoints (`backend/main.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/create_job` | Accept CSV files + hyperparams JSON, queue training task, return `{run_id, cancel_token}` |
| GET  | `/check_status/{run_id}` | Poll job status from DB |
| POST | `/cancel_job/{run_id}` | Cancel job (requires cancel token, SHA-256 verified) |
| GET  | `/metrics/{run_id}` | Live per-epoch metrics JSON (polled by frontend) |
| GET  | `/download_embedding/{run_id}` | Return `.pkl` embedding file |
| GET  | `/download_model/{run_id}` | Return `.pt` model weights |
| POST | `/run_inference/{source_run_id}` | Queue inference job against a trained model |
| GET  | `/download_results/{run_id}` | Return inference results CSV |
| GET  | `/jobs` | List all jobs (training + inference) with final metrics |
| GET  | `/health` | Health check |

## Frontend Pages (`frontend/`)

- `app.py` — Entry point, `st.navigation()` with grouped sidebar:
  - **""** (Home): `home.py`
  - **Model Building**: `build.py`, `dti.py`, `subcellular.py`, `rna_prot.py`, `protein_function.py`, `prot_dna.py`
  - **Tools**: `inference.py`, `check_results.py`, `job_status.py`
- `build.py` — PPI training submission: CSV upload → column mapping → validation → hyperparameter config + live model architecture visualization → submit
- `check_results.py` — Monitor a specific run: status badge, progress bar, loss/accuracy charts, final metrics, download buttons, cancel form
- `job_status.py` — Table of all submitted jobs with detail panel and downloads
- `inference.py` — Run inference on new pairs using a completed training run
- `home.py` — Platform landing page with task cards (Available vs Coming Soon)
- `dti.py`, `subcellular.py`, `rna_prot.py`, `protein_function.py`, `prot_dna.py` — Coming soon pages

## Key Implementation Details

- **ESM2 model**: `ESM2_35M` (12-layer, 480-dim embeddings) pre-downloaded into `backend/ESM2_35M/`. `ESM2_8M/` also present.
- **Sliding window**: Sequences > 1022 residues use a sliding window (stride 512) in `esm_embed.py`.
- **Pair representation**: `concat` / `product` / `diff` / `all` (recommended). `all` = 480×4 = 1920-dim input.
- **MLP classifier**: `backend/model_build/ppi_classifier.py` — configurable depth/width, ReLU + Dropout(0.3), class-weighted CrossEntropyLoss, per-epoch JSON metrics, early stopping on val loss.
- **Inference**: `backend/model_build/ppi_infer.py` — loads saved model, embeds new sequences, returns probabilities.
- **Celery**: Redis broker + result backend. 4-hour soft/hard time limits. `SoftTimeLimitExceeded` marks job as failed gracefully.
- **Cancel token**: `secrets.token_urlsafe(32)` shown once to user; SHA-256 hash stored in DB. Cancel verified by re-hashing and comparing. Celery task revoked via `celery.control.revoke(terminate=True)`.
- **NaN sanitization**: `_safe(v)` helper in both `ppi_classifier.py` and `main.py` converts numpy floats and replaces NaN/Inf with `None` before JSON serialization.
- **DB migration**: `startup()` in `main.py` runs `ALTER TABLE jobs ADD COLUMN IF NOT EXISTS` for all new columns — safe to run on existing DBs.
- **Column mapping**: Frontend accepts any CSV column names; user maps to proteinA/proteinB/label via dropdowns; CSV is renamed before sending to backend.
- **GPU**: Backend and Celery worker use `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime` with NVIDIA GPU access via `deploy.resources.reservations.devices`.

## Database Model (`backend/models.py`)

`Job` table columns: `run_id`, `status`, `job_type` (train/inference), `input_sequence`, `hyperparams`, `model_path`, `metrics`, `source_run_id`, `cancel_token_hash`, `celery_task_id`, `result`, `created_at`

## Configuration

All configuration is hardcoded or via Docker Compose environment variables — no `.env` file:
- `BACKEND_URL` env var used by frontend (default: `http://backend:8005`)
- Database credentials hardcoded in `backend/database.py`
- ESM2 model paths relative within container (`./ESM2_35M`)
- `backend/saved_models/` and `backend/ESM2_*` are git-ignored

## Nginx

Production: `web3.compbiosysnbu.in` proxied via `/etc/nginx/sites-available/web3`
- Frontend: `proxy_pass http://127.0.0.1:8502/`
- Backend API: `proxy_pass http://127.0.0.1:8006/`

## Common Gotchas

- Always use `docker compose` (v2), never `docker-compose` (v1) — v1 has bugs with newer images and GPU syntax.
- After changing `backend/tasks.py` or `backend/model_build/`, rebuild **both** `backend` and `celery-worker`.
- FastAPI's JSON encoder rejects `float('nan')` — always pass through `_safe()` before returning metrics.
- The `cancel_token` is returned only once at job creation. If frontend session is lost, user cannot cancel (by design — no auth system).


# rebuilding
- you dont need to run docker compose , i will run it myself