import hashlib
import json
import math
import os
import re
import secrets
import shutil
import uuid
from datetime import datetime, timedelta, timezone

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _valid_run_id(run_id: str) -> bool:
    return bool(_RUN_ID_RE.match(run_id or ""))


def _safe(v):
    """Return None for NaN/Inf floats; otherwise pass through."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return v

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import text
from typing import List

from database import Base, engine, SessionLocal
from models import Job
from tasks import (
    train_ppi_model, run_ppi_inference,
    train_dti_model, run_dti_inference_task,
    train_rpi_model, run_rpi_inference_task,
    train_pdi_model, run_pdi_inference_task,
)

MODELS_DIR = "/app/saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup: create tables + migrate new columns
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

    new_cols = [
        ("job_type",          "VARCHAR"),
        ("hyperparams",       "TEXT"),
        ("model_path",        "VARCHAR"),
        ("metrics",           "TEXT"),
        ("source_run_id",     "VARCHAR"),
        ("cancel_token_hash", "TEXT"),
        ("celery_task_id",    "VARCHAR"),
        ("result",            "TEXT"),
    ]
    with engine.connect() as conn:
        for col, dtype in new_cols:
            try:
                conn.execute(text(
                    f"ALTER TABLE jobs ADD COLUMN IF NOT EXISTS {col} {dtype}"
                ))
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"[migration] {col}: {e}", flush=True)

    _run_cleanup()


def _run_cleanup(failed_ttl_days: int = 7, completed_ttl_days: int = 30):
    """
    Artifact + DB cleanup (runs on every backend startup).

    - running  stuck > 5 h  → mark failed, delete dir
    - queued   stuck > 1 d  → mark failed, delete dir
    - failed / cancelled    → delete dir immediately; purge DB record after 7 days
    - completed             → delete dir + DB record after 30 days
    """
    db  = SessionLocal()
    now = datetime.now(timezone.utc)
    failed_cutoff    = now - timedelta(days=failed_ttl_days)
    completed_cutoff = now - timedelta(days=completed_ttl_days)
    stale_running_cutoff = now - timedelta(hours=5)   # task hard limit is 4 h + 1 h grace
    stale_queued_cutoff  = now - timedelta(days=1)
    deleted_dirs, deleted_rows, marked_stale = 0, 0, 0

    try:
        jobs = db.query(Job).all()
        for job in jobs:
            run_dir = os.path.join(MODELS_DIR, job.run_id)

            created = job.created_at
            if created is not None and created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)

            # ── Mark stale running/queued jobs as failed ──────────────────────
            if job.status == "running" and created is not None and created < stale_running_cutoff:
                job.status = "failed"
                job.result = "Job marked stale: still running after 5 hours (worker likely lost)."
                marked_stale += 1
            elif job.status == "queued" and created is not None and created < stale_queued_cutoff:
                job.status = "failed"
                job.result = "Job marked stale: not picked up by a worker within 24 hours."
                marked_stale += 1

            # ── Failed / cancelled ────────────────────────────────────────────
            if job.status in ("failed", "cancelled"):
                if os.path.isdir(run_dir):
                    try:
                        shutil.rmtree(run_dir)
                        deleted_dirs += 1
                    except Exception as exc:
                        print(f"[cleanup] rmtree {run_dir}: {exc}", flush=True)
                if created is not None and created < failed_cutoff:
                    db.delete(job)
                    deleted_rows += 1

            # ── Completed ─────────────────────────────────────────────────────
            elif job.status == "completed":
                if created is not None and created < completed_cutoff:
                    if os.path.isdir(run_dir):
                        try:
                            shutil.rmtree(run_dir)
                            deleted_dirs += 1
                        except Exception as exc:
                            print(f"[cleanup] TTL rmtree {run_dir}: {exc}", flush=True)
                    db.delete(job)
                    deleted_rows += 1

        db.commit()
    except Exception as exc:
        print(f"[cleanup] Unexpected error: {exc}", flush=True)
        db.rollback()
    finally:
        db.close()

    print(
        f"[cleanup] Startup sweep — stale→failed: {marked_stale}, "
        f"dirs removed: {deleted_dirs}, "
        f"DB rows purged (failed>{failed_ttl_days}d / completed>{completed_ttl_days}d): {deleted_rows}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_dir(run_id: str) -> str:
    path = os.path.join(MODELS_DIR, run_id)
    os.makedirs(path, exist_ok=True)
    return path


def _save_uploaded_files(files: List[UploadFile], run_dir: str, run_id: str) -> list:
    paths = []
    for file in files:
        safe_name = os.path.basename(file.filename or "upload.csv")
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", safe_name) or "upload.csv"
        dest = os.path.join(run_dir, f"{run_id}_{safe_name}")
        with open(dest, "wb") as f:
            f.write(file.file.read())
        paths.append(dest)
    return paths


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}



# ---------------------------------------------------------------------------
# POST /create_job  — queue a training job
# ---------------------------------------------------------------------------

@app.post("/create_job")
async def create_job(
    files:       List[UploadFile] = File(...),
    hyperparams: str              = Form("{}"),   # JSON string
):
    run_id  = str(uuid.uuid4())[:8]
    run_dir = _run_dir(run_id)
    paths   = _save_uploaded_files(files, run_dir, run_id)

    # validate hyperparams JSON
    try:
        hp = json.loads(hyperparams)
    except Exception:
        hp = {}

    cancel_token      = secrets.token_urlsafe(32)          # shown once to user
    cancel_token_hash = hashlib.sha256(cancel_token.encode()).hexdigest()

    db  = SessionLocal()
    try:
        job = Job(
            run_id            = run_id,
            status            = "queued",
            job_type          = "train",
            input_sequence    = json.dumps(paths),
            hyperparams       = json.dumps(hp),
            cancel_token_hash = cancel_token_hash,
        )
        db.add(job)
        db.commit()
    finally:
        db.close()

    task_type = hp.get("task_type", "ppi")
    if task_type == "dti":
        task = train_dti_model.delay(run_id, paths, json.dumps(hp))
    elif task_type == "rpi":
        task = train_rpi_model.delay(run_id, paths, json.dumps(hp))
    elif task_type == "pdi":
        task = train_pdi_model.delay(run_id, paths, json.dumps(hp))
    else:
        task = train_ppi_model.delay(run_id, paths, json.dumps(hp))

    # store celery task id so we can revoke it later
    db  = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        job.celery_task_id = task.id
        db.commit()
    finally:
        db.close()

    return {"run_id": run_id, "cancel_token": cancel_token}


# ---------------------------------------------------------------------------
# GET /check_status/{run_id}
# ---------------------------------------------------------------------------

@app.get("/check_status/{run_id}")
def check_status(run_id: str):
    db  = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        if not job:
            return {"error": "invalid run_id"}
        return {
            "run_id":      job.run_id,
            "status":      job.status,
            "job_type":    job.job_type,
            "result":      job.result,
            "hyperparams": json.loads(job.hyperparams) if job.hyperparams else {},
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# POST /cancel_job/{run_id}  — cancel with token verification
# ---------------------------------------------------------------------------

from fastapi import Body

@app.post("/cancel_job/{run_id}")
def cancel_job(run_id: str, cancel_token: str = Body(..., embed=True)):
    db  = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        if not job:
            return JSONResponse({"error": "run_id not found"}, status_code=404)

        if job.status in ("completed", "failed", "cancelled"):
            return JSONResponse({"error": f"Job is already {job.status} — cannot cancel."}, status_code=400)

        # verify token
        provided_hash = hashlib.sha256(cancel_token.encode()).hexdigest()
        if provided_hash != job.cancel_token_hash:
            return JSONResponse({"error": "Invalid cancel token."}, status_code=403)

        celery_task_id = job.celery_task_id
    finally:
        db.close()

    # revoke celery task
    if celery_task_id:
        from tasks import celery as celery_app
        celery_app.control.revoke(celery_task_id, terminate=True, signal="SIGTERM")

    db  = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        job.status = "cancelled"
        job.result = "Cancelled by user."
        db.commit()
    finally:
        db.close()

    return {"status": "cancelled", "run_id": run_id}


# ---------------------------------------------------------------------------
# GET /metrics/{run_id}  — live training metrics (polled by frontend)
# ---------------------------------------------------------------------------

@app.get("/metrics/{run_id}")
def get_metrics(run_id: str):
    if not _valid_run_id(run_id):
        return JSONResponse({"error": "invalid run_id"}, status_code=400)
    metrics_path = os.path.join(MODELS_DIR, run_id, f"metrics_{run_id}.json")
    if not os.path.exists(metrics_path):
        return {"status": "pending", "epoch": 0, "total_epochs": 0, "history": {}}

    with open(metrics_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# GET /download_embedding/{run_id}
# ---------------------------------------------------------------------------

@app.get("/download_embedding/{run_id}")
def download_embedding(run_id: str):
    if not _valid_run_id(run_id):
        return JSONResponse({"error": "invalid run_id"}, status_code=400)
    path = os.path.join(MODELS_DIR, run_id, f"embedding_{run_id}.pkl")
    if not os.path.exists(path):
        return JSONResponse({"error": "embedding not found"}, status_code=404)
    return FileResponse(path, media_type="application/octet-stream",
                        filename=f"embedding_{run_id}.pkl")


# ---------------------------------------------------------------------------
# GET /download_model/{run_id}
# ---------------------------------------------------------------------------

@app.get("/download_model/{run_id}")
def download_model(run_id: str):
    if not _valid_run_id(run_id):
        return JSONResponse({"error": "invalid run_id"}, status_code=400)
    path = os.path.join(MODELS_DIR, run_id, f"model_{run_id}.pt")
    if not os.path.exists(path):
        return JSONResponse({"error": "model not found"}, status_code=404)
    return FileResponse(path, media_type="application/octet-stream",
                        filename=f"model_{run_id}.pt")


# ---------------------------------------------------------------------------
# POST /run_inference/{source_run_id}  — queue an inference job
# ---------------------------------------------------------------------------

@app.post("/run_inference/{source_run_id}")
async def create_inference_job(
    source_run_id: str,
    files: List[UploadFile] = File(...),
):
    if not _valid_run_id(source_run_id):
        return JSONResponse({"error": "invalid source_run_id"}, status_code=400)
    # verify source job exists and is a completed training run
    db  = SessionLocal()
    try:
        src = db.query(Job).filter(Job.run_id == source_run_id).first()
    finally:
        db.close()

    if not src:
        return JSONResponse({"error": "source run_id not found"}, status_code=404)
    if src.status != "completed":
        return JSONResponse({"error": "source job not completed yet"}, status_code=400)
    if src.job_type != "train":
        return JSONResponse({"error": "source job is not a training run"}, status_code=400)

    # Determine task type from source job hyperparams
    src_hp = {}
    if src.hyperparams:
        try:
            src_hp = json.loads(src.hyperparams)
        except Exception:
            pass
    src_task_type = src_hp.get("task_type", "ppi")

    run_id  = str(uuid.uuid4())[:8]
    run_dir = _run_dir(run_id)
    paths   = _save_uploaded_files(files, run_dir, run_id)

    db  = SessionLocal()
    try:
        job = Job(
            run_id         = run_id,
            status         = "queued",
            job_type       = "inference",
            input_sequence = json.dumps(paths),
            source_run_id  = source_run_id,
        )
        db.add(job)
        db.commit()
    finally:
        db.close()

    if src_task_type == "dti":
        run_dti_inference_task.delay(run_id, source_run_id, paths)
    elif src_task_type == "rpi":
        run_rpi_inference_task.delay(run_id, source_run_id, paths)
    elif src_task_type == "pdi":
        run_pdi_inference_task.delay(run_id, source_run_id, paths)
    else:
        run_ppi_inference.delay(run_id, source_run_id, paths)

    return {"run_id": run_id}


# ---------------------------------------------------------------------------
# GET /download_results/{run_id}  — inference results CSV
# ---------------------------------------------------------------------------

@app.get("/download_results/{run_id}")
def download_results(run_id: str):
    if not _valid_run_id(run_id):
        return JSONResponse({"error": "invalid run_id"}, status_code=400)
    path = os.path.join(MODELS_DIR, run_id, f"results_{run_id}.csv")
    if not os.path.exists(path):
        return JSONResponse({"error": "results not found"}, status_code=404)
    return FileResponse(path, media_type="text/csv",
                        filename=f"ppi_results_{run_id}.csv")


# ---------------------------------------------------------------------------
# GET /inference_metrics/{run_id}  — rich metrics for the inference dashboard
# ---------------------------------------------------------------------------

@app.get("/inference_metrics/{run_id}")
def get_inference_metrics(run_id: str):
    if not _valid_run_id(run_id):
        return JSONResponse({"error": "invalid run_id"}, status_code=400)
    """
    Return predicted probabilities, optional ground-truth labels, and
    pre-computed aggregate metrics for the inference dashboard.
    Written by run_ppi_inference alongside the results CSV.
    """
    metrics_path = os.path.join(MODELS_DIR, run_id, f"infer_metrics_{run_id}.json")
    if not os.path.exists(metrics_path):
        # Graceful fallback: derive from results CSV if metrics file absent
        csv_path = os.path.join(MODELS_DIR, run_id, f"results_{run_id}.csv")
        if not os.path.exists(csv_path):
            return JSONResponse({"error": "not found"}, status_code=404)
        import pandas as pd
        df = pd.read_csv(csv_path)
        probs = df["probability"].dropna().tolist() if "probability" in df.columns else []
        return {
            "has_labels":    False,
            "probabilities": probs,
            "labels":        [],
        }

    with open(metrics_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers — dataset stats & UMAP
# ---------------------------------------------------------------------------

def _histogram(values: list, bins: int = 25) -> dict:
    import numpy as np
    arr = np.array(values, dtype=float)
    counts, edges = np.histogram(arr, bins=bins)
    return {"counts": counts.tolist(), "bins": [round(float(v), 1) for v in edges]}


def _reconstruct_val_set(df) -> set:
    """Reproduce the exact validation indices used during training (random_state=42)."""
    from sklearn.model_selection import train_test_split
    labels = df["label"].astype(int).tolist() if "label" in df.columns else [0] * len(df)
    idx    = list(range(len(labels)))
    try:
        _, va_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)
    except ValueError:
        _, va_idx = train_test_split(idx, test_size=0.2, random_state=42)
    return set(va_idx)


def _build_pair_embeddings(run_id: str, run_dir: str, task_type: str, df) -> tuple:
    """Returns (X, labels, orig_df_indices) — orig_df_indices[i] is the df row for X[i]."""
    import pickle
    import numpy as np

    def _load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def _label(row):
        try:
            v = row.get("label", "")
            return int(v) if str(v) not in ("nan", "", "None") else -1
        except Exception:
            return -1

    rows, labels, orig_idx = [], [], []

    if task_type == "ppi":
        emb = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        for i, (_, row) in enumerate(df.iterrows()):
            a = str(row.get("proteinA", "")).strip().upper()
            b = str(row.get("proteinB", "")).strip().upper()
            if a in emb and b in emb:
                rows.append(np.concatenate([emb[a], emb[b]]))
                labels.append(_label(row))
                orig_idx.append(i)

    elif task_type == "dti":
        chem = _load(os.path.join(run_dir, f"chem_embedding_{run_id}.pkl"))
        esm  = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        for i, (_, row) in enumerate(df.iterrows()):
            s = str(row.get("smiles", "")).strip()
            p = str(row.get("sequence", "")).strip().upper()
            if s in chem and p in esm:
                rows.append(np.concatenate([chem[s], esm[p]]))
                labels.append(_label(row))
                orig_idx.append(i)

    elif task_type == "rpi":
        rna = _load(os.path.join(run_dir, f"rna_embedding_{run_id}.pkl"))
        esm = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        for i, (_, row) in enumerate(df.iterrows()):
            r = str(row.get("rna_sequence", "")).strip().upper().replace("T", "U")
            p = str(row.get("protein_sequence", "")).strip().upper()
            if r in rna and p in esm:
                rows.append(np.concatenate([rna[r], esm[p]]))
                labels.append(_label(row))
                orig_idx.append(i)

    elif task_type == "pdi":
        dna = _load(os.path.join(run_dir, f"dna_embedding_{run_id}.pkl"))
        esm = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        for i, (_, row) in enumerate(df.iterrows()):
            d = str(row.get("dna_sequence", "")).strip().upper()
            p = str(row.get("protein_sequence", "")).strip().upper()
            if d in dna and p in esm:
                rows.append(np.concatenate([dna[d], esm[p]]))
                labels.append(_label(row))
                orig_idx.append(i)

    if not rows:
        import numpy as np
        return np.array([]), [], []
    return np.array(rows, dtype=np.float32), labels, orig_idx


def _run_umap(X):
    import numpy as np
    import umap as _umap

    n, d = X.shape
    # PCA pre-reduction only for datasets large enough to benefit
    if n >= 100:
        from sklearn.decomposition import PCA
        n_pca = min(50, d, n - 1)
        X = PCA(n_components=n_pca, random_state=42).fit_transform(X)

    n_neighbors = min(15, n - 1)
    return _umap.UMAP(
        n_components=2, random_state=42,
        n_neighbors=n_neighbors, min_dist=0.1,
    ).fit_transform(X)


def _build_model_inputs(run_id: str, run_dir: str, task_type: str, df, pair_mode: str) -> tuple:
    """Returns (X, labels, orig_df_indices) using the exact pair_mode from training."""
    import pickle
    import numpy as np
    import torch

    def _load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def _label(row):
        try:
            v = row.get("label", "")
            return int(v) if str(v) not in ("nan", "", "None") else -1
        except Exception:
            return -1

    rows, labels, orig_idx = [], [], []

    if task_type == "ppi":
        emb = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        for i, (_, row) in enumerate(df.iterrows()):
            a = str(row.get("proteinA", "")).strip().upper()
            b = str(row.get("proteinB", "")).strip().upper()
            if a in emb and b in emb:
                eA = torch.tensor(np.array(emb[a], dtype=np.float32))
                eB = torch.tensor(np.array(emb[b], dtype=np.float32))
                if pair_mode == "product":
                    vec = (eA * eB).numpy()
                elif pair_mode == "diff":
                    vec = (eA - eB).abs().numpy()
                elif pair_mode == "all":
                    vec = torch.cat([eA, eB, eA * eB, (eA - eB).abs()]).numpy()
                else:
                    vec = torch.cat([eA, eB]).numpy()
                rows.append(vec)
                labels.append(_label(row))
                orig_idx.append(i)

    elif task_type == "dti":
        chem = _load(os.path.join(run_dir, f"chem_embedding_{run_id}.pkl"))
        esm  = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        for i, (_, row) in enumerate(df.iterrows()):
            s = str(row.get("smiles", "")).strip()
            p = str(row.get("sequence", "")).strip().upper()
            if s in chem and p in esm:
                rows.append(np.concatenate([chem[s], esm[p]]))
                labels.append(_label(row))
                orig_idx.append(i)

    elif task_type == "rpi":
        rna = _load(os.path.join(run_dir, f"rna_embedding_{run_id}.pkl"))
        esm = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        for i, (_, row) in enumerate(df.iterrows()):
            r = str(row.get("rna_sequence", "")).strip().upper().replace("T", "U")
            p = str(row.get("protein_sequence", "")).strip().upper()
            if r in rna and p in esm:
                rows.append(np.concatenate([rna[r], esm[p]]))
                labels.append(_label(row))
                orig_idx.append(i)

    elif task_type == "pdi":
        dna = _load(os.path.join(run_dir, f"dna_embedding_{run_id}.pkl"))
        esm = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        for i, (_, row) in enumerate(df.iterrows()):
            d = str(row.get("dna_sequence", "")).strip().upper()
            p = str(row.get("protein_sequence", "")).strip().upper()
            if d in dna and p in esm:
                rows.append(np.concatenate([dna[d], esm[p]]))
                labels.append(_label(row))
                orig_idx.append(i)

    if not rows:
        return np.array([]), [], []
    return np.array(rows, dtype=np.float32), labels, orig_idx


# ---------------------------------------------------------------------------
# GET /dataset_stats/{run_id}
# ---------------------------------------------------------------------------

@app.get("/dataset_stats/{run_id}")
def get_dataset_stats(run_id: str):
    if not _valid_run_id(run_id):
        return JSONResponse({"error": "invalid run_id"}, status_code=400)

    cache = os.path.join(MODELS_DIR, run_id, f"dataset_stats_{run_id}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        if not job:
            return JSONResponse({"error": "not found"}, status_code=404)
        hp          = json.loads(job.hyperparams) if job.hyperparams else {}
        task_type   = hp.get("task_type", "ppi")
        input_files = json.loads(job.input_sequence) if job.input_sequence else []
    finally:
        db.close()

    existing = [p for p in input_files if os.path.exists(p)]
    if not existing:
        return JSONResponse({"error": "data files not available"}, status_code=404)

    try:
        import pandas as pd
        df = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    stats: dict = {"task_type": task_type, "n_total": len(df)}

    if "label" in df.columns:
        vc = df["label"].value_counts().to_dict()
        stats["n_positive"]    = int(vc.get(1, 0))
        stats["n_negative"]    = int(vc.get(0, 0))
        stats["positive_rate"] = round(stats["n_positive"] / max(len(df), 1), 4)

    col_labels = {
        "ppi": [("proteinA", "Protein A"), ("proteinB", "Protein B")],
        "dti": [("smiles", "SMILES"), ("sequence", "Protein")],
        "rpi": [("rna_sequence", "RNA"), ("protein_sequence", "Protein")],
        "pdi": [("dna_sequence", "DNA"), ("protein_sequence", "Protein")],
    }
    hists = {}
    for col, label in col_labels.get(task_type, []):
        if col in df.columns:
            hists[label] = _histogram(df[col].astype(str).str.len().tolist())
    stats["length_hists"] = hists

    with open(cache, "w") as f:
        json.dump(stats, f)
    return stats


# ---------------------------------------------------------------------------
# GET /umap_data/{run_id}
# ---------------------------------------------------------------------------

@app.get("/umap_data/{run_id}")
def get_umap_data(run_id: str):
    if not _valid_run_id(run_id):
        return JSONResponse({"error": "invalid run_id"}, status_code=400)

    # v2 cache includes split info
    cache = os.path.join(MODELS_DIR, run_id, f"emb_umap_{run_id}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        if not job:
            return JSONResponse({"error": "not found"}, status_code=404)
        if job.status != "completed":
            return JSONResponse({"error": "job not completed"}, status_code=400)
        hp          = json.loads(job.hyperparams) if job.hyperparams else {}
        task_type   = hp.get("task_type", "ppi")
        input_files = json.loads(job.input_sequence) if job.input_sequence else []
    finally:
        db.close()

    existing = [p for p in input_files if os.path.exists(p)]
    if not existing:
        return JSONResponse({"error": "data files not available"}, status_code=404)

    try:
        import pandas as pd
        df = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    run_dir = os.path.join(MODELS_DIR, run_id)
    try:
        X, labels, orig_indices = _build_pair_embeddings(run_id, run_dir, task_type, df)
    except Exception as e:
        return JSONResponse({"error": f"embedding load failed: {e}"}, status_code=500)

    if len(X) < 10:
        return JSONResponse(
            {"error": f"too few samples ({len(X)}) for UMAP — need ≥ 10"},
            status_code=400,
        )

    try:
        import numpy as np

        val_set = _reconstruct_val_set(df)
        splits  = ["val" if orig_indices[i] in val_set else "train" for i in range(len(X))]

        MAX_N = 5000
        if len(X) > MAX_N:
            rng   = np.random.default_rng(42)
            ratio = MAX_N / len(X)
            # stratify by (label, split) to keep proportions
            by_group: dict = {}
            for i, (lbl, spl) in enumerate(zip(labels, splits)):
                by_group.setdefault((lbl, spl), []).append(i)
            chosen: list = []
            for idxs in by_group.values():
                k = max(1, int(len(idxs) * ratio))
                chosen.extend(rng.choice(idxs, min(k, len(idxs)), replace=False).tolist())
            chosen  = chosen[:MAX_N]
            X       = X[chosen]
            labels  = [labels[i]  for i in chosen]
            splits  = [splits[i]  for i in chosen]

        coords = _run_umap(X)
    except Exception as e:
        return JSONResponse({"error": f"UMAP failed: {e}"}, status_code=500)

    result = {
        "x":        [round(float(v), 4) for v in coords[:, 0]],
        "y":        [round(float(v), 4) for v in coords[:, 1]],
        "labels":   labels,
        "splits":   splits,
        "n_samples": len(labels),
    }
    with open(cache, "w") as f:
        json.dump(result, f)
    return result


# ---------------------------------------------------------------------------
# GET /model_umap/{run_id}  — penultimate-layer UMAP colored by TP/TN/FP/FN
# ---------------------------------------------------------------------------

@app.get("/model_umap/{run_id}")
def get_model_umap(run_id: str):
    if not _valid_run_id(run_id):
        return JSONResponse({"error": "invalid run_id"}, status_code=400)

    cache = os.path.join(MODELS_DIR, run_id, f"model_umap_{run_id}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        if not job:
            return JSONResponse({"error": "not found"}, status_code=404)
        if job.status != "completed":
            return JSONResponse({"error": "job not completed"}, status_code=400)
        hp          = json.loads(job.hyperparams) if job.hyperparams else {}
        task_type   = hp.get("task_type", "ppi")
        input_files = json.loads(job.input_sequence) if job.input_sequence else []
    finally:
        db.close()

    model_path = os.path.join(MODELS_DIR, run_id, f"model_{run_id}.pt")
    if not os.path.exists(model_path):
        return JSONResponse({"error": "model file not found"}, status_code=404)

    existing = [p for p in input_files if os.path.exists(p)]
    if not existing:
        return JSONResponse({"error": "data files not available"}, status_code=404)

    try:
        import pandas as pd
        df = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    try:
        import torch
        import numpy as np
        from model_build.ppi_classifier import FlexiblePPIModel

        ckpt         = torch.load(model_path, map_location="cpu")
        input_dim    = ckpt["input_dim"]
        layer_configs = ckpt["layer_configs"]
        pair_mode    = ckpt.get("pair_mode", "concat")

        model = FlexiblePPIModel(input_dim, layer_configs)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        run_dir = os.path.join(MODELS_DIR, run_id)
        X, true_labels, orig_indices = _build_model_inputs(run_id, run_dir, task_type, df, pair_mode)

        # keep only validation samples
        val_set    = _reconstruct_val_set(df)
        val_mask   = [i for i, oi in enumerate(orig_indices) if oi in val_set]
        X          = X[val_mask]
        true_labels = [true_labels[i] for i in val_mask]
    except Exception as e:
        return JSONResponse({"error": f"model load failed: {e}"}, status_code=500)

    if len(X) < 10:
        return JSONResponse(
            {"error": f"too few validation samples ({len(X)}) for UMAP — need ≥ 10"},
            status_code=400,
        )

    try:
        import numpy as np
        import torch

        # stratified cap at 5000
        MAX_N = 5000
        if len(X) > MAX_N:
            rng   = np.random.default_rng(42)
            ratio = MAX_N / len(X)
            by_label: dict = {}
            for i, lbl in enumerate(true_labels):
                by_label.setdefault(lbl, []).append(i)
            chosen: list = []
            for idxs in by_label.values():
                k = max(1, int(len(idxs) * ratio))
                chosen.extend(rng.choice(idxs, min(k, len(idxs)), replace=False).tolist())
            chosen = chosen[:MAX_N]
            X           = X[chosen]
            true_labels = [true_labels[i] for i in chosen]

        # hook into output layer to capture penultimate activations
        penultimate: list = []
        def _hook(module, inp, out):
            penultimate.append(inp[0].detach().cpu().numpy())
        handle = model.output.register_forward_hook(_hook)

        probs_all: list = []
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, len(X_tensor), 256):
                batch  = X_tensor[i : i + 256]
                logits = model(batch)
                probs_all.extend(torch.sigmoid(logits).numpy().tolist())

        handle.remove()
        penult_feats = np.vstack(penultimate)
        pred_labels  = [1 if p >= 0.5 else 0 for p in probs_all]

        categories = []
        for tl, pl in zip(true_labels, pred_labels):
            if tl == -1:
                categories.append("Unknown")
            elif tl == 1 and pl == 1:
                categories.append("TP")
            elif tl == 0 and pl == 0:
                categories.append("TN")
            elif tl == 0 and pl == 1:
                categories.append("FP")
            else:
                categories.append("FN")

        coords = _run_umap(penult_feats)

        result = {
            "x":          [round(float(v), 4) for v in coords[:, 0]],
            "y":          [round(float(v), 4) for v in coords[:, 1]],
            "categories": categories,
            "n_samples":  len(categories),
        }
        with open(cache, "w") as f:
            json.dump(result, f)
        return result

    except Exception as e:
        return JSONResponse({"error": f"model UMAP failed: {e}"}, status_code=500)


# ---------------------------------------------------------------------------
# GET /jobs  — list all jobs
# ---------------------------------------------------------------------------

@app.get("/jobs")
def list_jobs():
    db   = SessionLocal()
    try:
        jobs = db.query(Job).all()
        out  = []
        for j in jobs:
            metrics = {}
            if j.metrics:
                try:
                    raw = json.loads(j.metrics)
                    metrics = raw.get("final", {})
                except Exception:
                    pass

            hp_raw = {}
            if j.hyperparams:
                try:
                    hp_raw = json.loads(j.hyperparams)
                except Exception:
                    pass

            out.append({
                "run_id":        j.run_id,
                "status":        j.status,
                "job_type":      j.job_type or "train",
                "task_type":     hp_raw.get("task_type", "ppi"),
                "created_at":    j.created_at.isoformat() if j.created_at else None,
                "val_acc":       _safe(metrics.get("val_acc")),
                "auroc":         _safe(metrics.get("auroc")),
                "f1":            _safe(metrics.get("f1")),
                "source_run_id": j.source_run_id,
                # model architecture info
                "esm_model":     hp_raw.get("esm_model", "—"),
                "esm_dim":       hp_raw.get("esm_dim"),
                "chem_model":    hp_raw.get("chem_model", "—"),
                "chem_dim":      hp_raw.get("chem_dim"),
                "rna_model":     hp_raw.get("rna_model", "—"),
                "rna_dim":       hp_raw.get("rna_dim"),
                "dna_model":     hp_raw.get("dna_model", "—"),
                "dna_dim":       hp_raw.get("dna_dim"),
                "layer_configs": hp_raw.get("layer_configs", []),
                "epochs":        hp_raw.get("epochs"),
                "train_split":   hp_raw.get("train_split"),
            })
        return out
    finally:
        db.close()
