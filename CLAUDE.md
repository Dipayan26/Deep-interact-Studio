# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Protein-Protein Interaction (PPI) Prediction Platform** — a web tool that lets users upload protein sequence CSVs, generate ESM2 embeddings via async GPU workers, and download results. The stack is fully containerized.

## Running the Project

```bash
# Start all services (recommended)
docker-compose up --build

# Rebuild without cache
docker-compose build --no-cache

# Run specific service
docker-compose up backend
docker-compose up frontend
```

Service URLs:
- Frontend (Streamlit): http://localhost:8501
- Backend (FastAPI): http://localhost:8005
- PostgreSQL: localhost:5432 (db: `ppi_jobs`, user: `ppi_user`, pass: `ppi_pass`)
- Redis: localhost:6379

## Architecture

Five Docker services communicate as follows:

```
Streamlit (8501) → FastAPI (8005) → Redis → Celery Worker (GPU)
                        ↕
                   PostgreSQL (5432)
```

**Key data flow:**
1. User uploads CSV via Streamlit (`frontend/`)
2. FastAPI (`backend/main.py`) saves the file, creates a DB job record, and enqueues a Celery task
3. Celery worker (`backend/tasks.py`) runs ESM2 protein embedding (`backend/model_build/esm_embed.py`) on GPU
4. Embeddings saved as pickle to `/app/saved_models/{run_id}/`
5. Job status updated in PostgreSQL; user can poll and download via Streamlit

**Backend API endpoints** (`backend/main.py`):
- `POST /create_job` — Accept CSV files, queue embedding task
- `GET /check_status/{run_id}` — Poll job status from DB
- `GET /download_embedding/{run_id}` — Return pickled embedding file
- `GET /jobs` — List all jobs
- `GET /health` — Health check

## Key Implementation Details

- **ESM2 model**: `ESM2_35M` (12-layer, 35M parameter protein language model) pre-downloaded into `backend/ESM2_35M/`. The `ESM2_8M/` directory also exists for the smaller variant.
- **Sliding window**: For sequences longer than 1022 tokens, `esm_embed.py` uses a sliding window approach with overlap.
- **Celery**: Uses Redis as both broker and result backend. Worker defined in `backend/celery_worker.py`, tasks in `backend/tasks.py`.
- **Database models**: `backend/models.py` defines the `Job` SQLAlchemy model; `backend/database.py` connects to PostgreSQL.
- **Frontend pages**: `frontend/app.py` is the entry point; it imports `home.py`, `build.py`, and `job_status.py` as page modules.
- **GPU**: Backend and Celery worker containers use `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime` and require NVIDIA GPU access.

## Local Development (without Docker)

For backend:
```bash
cd backend
pip install -r requirements.txt
# Requires running Redis and PostgreSQL (update DATABASE_URL in database.py if needed)
uvicorn main:app --host 0.0.0.0 --port 8005 --reload

# Run Celery worker separately
celery -A celery_worker worker --loglevel=info
```

For frontend:
```bash
cd frontend
pip install -r requirements.txt
BACKEND_URL=http://localhost:8005 streamlit run app.py
```

## Configuration

All configuration is hardcoded or set via Docker Compose environment variables — there is no `.env` file:
- `BACKEND_URL` env var controls the backend URL used by the frontend (default: `http://backend:8005`)
- Database credentials are hardcoded in `backend/database.py`
- ESM2 model paths are relative within the container (`./ESM2_35M`)
- `backend/saved_models/` and `backend/ESM2_*` directories are git-ignored
