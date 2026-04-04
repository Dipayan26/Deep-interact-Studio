# Deep-Prot Studio

An open-source, GPU-accelerated platform for training and deploying deep learning models across biological sequence prediction tasks. Built around protein language model (PLM) embeddings, it enables researchers to build, evaluate, and apply predictive models — without writing ML code.

**Available:** PPI Prediction &nbsp;|&nbsp; **Coming Soon:** Drug–Target Interaction · Subcellular Localization · RNA–Protein Interaction · Protein Function (GO) · Protein–DNA Interaction

---

## Features

- ESM2 (35M) protein language model embeddings via GPU
- Configurable MLP classifier with pair representation options (concat / product / diff / all)
- Async training jobs via Celery + Redis — no page blocking
- Per-epoch live metrics (loss, accuracy) polled from the frontend
- Final metrics: Val Accuracy, AUROC, Precision, Recall, F1
- Inference on new protein pairs using a trained model
- Downloadable embeddings, model weights (`.pt`), and results (`.csv`)

---

## Architecture

```
Streamlit (8502) ──→ FastAPI (8006) ──→ Redis ──→ Celery Worker (GPU)
                          │
                     PostgreSQL (5433)
```

Five Docker services: `frontend`, `backend`, `celery-worker`, `redis`, `postgres`.

---

## Quick Start

**Requirements:** Docker, Docker Compose v2, NVIDIA GPU + drivers

```bash
git clone <repo-url>
cd ppi-webtool

docker compose up --build -d
```

Open **http://localhost:8502** in your browser.

---

## Input Format

### Training CSV

| proteinA | proteinB | label |
|---|---|---|
| MKTAYIAKQR... | MSHHWGYGKH... | 1 |
| MNIFEMLRID... | MKTAYIAKQR... | 0 |

- `proteinA`, `proteinB` — amino acid sequences (single-letter code, case-insensitive)
- `label` — `1` (interacting) or `0` (non-interacting)

### Inference CSV

Same format without the `label` column.

Sample files: [`examples/sample_train.csv`](examples/sample_train.csv), [`examples/sample_inference.csv`](examples/sample_inference.csv)

---

## Workflow

1. **Build page** — Upload training CSV, set hyperparameters, submit job
2. **Monitor** — Watch live loss/accuracy curves per epoch
3. **Download** — Retrieve embeddings (`.pkl`) and model weights (`.pt`)
4. **Inference page** — Select a trained model, upload new pairs, download predictions

---

## Hyperparameters

| Parameter | Options | Default |
|---|---|---|
| Pair representation | `all`, `concat`, `product`, `diff` | `all` |
| Hidden layer size | 128, 256, 512 | 256 |
| MLP depth | 2, 3 | 2 |
| Batch size | 32, 64, 128 | 64 |
| Epochs | 5–100 | 30 |
| Learning rate | 0.001, 0.0005, 0.0001 | 0.001 |

`all` = concatenation + element-wise product + absolute difference of the two protein embeddings (recommended).

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/create_job` | Submit training job |
| `GET` | `/check_status/{run_id}` | Poll job status |
| `GET` | `/metrics/{run_id}` | Live training metrics (JSON) |
| `GET` | `/download_embedding/{run_id}` | Download embeddings pickle |
| `GET` | `/download_model/{run_id}` | Download model weights |
| `POST` | `/run_inference/{source_run_id}` | Submit inference job |
| `GET` | `/download_results/{run_id}` | Download inference results CSV |
| `GET` | `/jobs` | List all jobs |
| `GET` | `/health` | Health check |

---

## Service Ports

| Service | Host Port |
|---|---|
| Frontend (Streamlit) | 8502 |
| Backend (FastAPI) | 8006 |
| PostgreSQL | 5433 |
| Redis | 6379 |

---

## Development

```bash
# Rebuild after code changes
docker compose build --no-cache frontend && docker compose up -d frontend
docker compose build --no-cache backend celery-worker && docker compose up -d backend celery-worker

# View logs
docker compose logs -f backend
docker compose logs -f celery-worker
```

### Without Docker

```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8006 --reload

# Celery worker (separate terminal)
celery -A celery_worker.celery worker --loglevel=info

# Frontend
cd frontend && pip install -r requirements.txt
BACKEND_URL=http://localhost:8006 streamlit run app.py
```

---

## Model

**ESM2 35M** (`esm2_t12_35M_UR50D`) — 12-layer transformer, 480-dimensional embeddings, trained on UniRef50. Embeddings are extracted from layer 12 and mean-pooled over residue positions. Sequences longer than 1022 residues are handled with a sliding window (stride 512).

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{ppi_webtool,
  author  = {Sarkar, Dipayan},
  title   = {PPI-Webtool: Sequence-based Protein–Protein Interaction Prediction Platform},
  year    = {2025},
  url     = {https://web3.compbiosysnbu.in}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
