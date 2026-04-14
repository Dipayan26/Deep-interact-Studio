import hashlib
import json
import math
import os
import secrets
import uuid


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
from tasks import train_ppi_model, run_ppi_inference, train_dti_model, run_dti_inference_task

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
        dest = os.path.join(run_dir, f"{run_id}_{file.filename}")
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
            "run_id":   job.run_id,
            "status":   job.status,
            "job_type": job.job_type,
            "result":   job.result,
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
    else:
        run_ppi_inference.delay(run_id, source_run_id, paths)

    return {"run_id": run_id}


# ---------------------------------------------------------------------------
# GET /download_results/{run_id}  — inference results CSV
# ---------------------------------------------------------------------------

@app.get("/download_results/{run_id}")
def download_results(run_id: str):
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
                "layer_configs": hp_raw.get("layer_configs", []),
                "epochs":        hp_raw.get("epochs"),
                "train_split":   hp_raw.get("train_split"),
            })
        return out
    finally:
        db.close()
