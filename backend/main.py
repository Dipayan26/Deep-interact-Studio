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
# GET /job_detail/{run_id}  — full job record including hyperparams
# ---------------------------------------------------------------------------

@app.get("/job_detail/{run_id}")
def job_detail(run_id: str):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        if not job:
            return JSONResponse({"error": "run_id not found"}, status_code=404)
        hp = {}
        if job.hyperparams:
            try:
                hp = json.loads(job.hyperparams)
            except Exception:
                pass
        return {
            "run_id":        job.run_id,
            "status":        job.status,
            "job_type":      job.job_type,
            "hyperparams":   hp,
            "source_run_id": job.source_run_id,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# GET /shap/{infer_run_id}  — SHAP feature-group importances for inference run
# ---------------------------------------------------------------------------

@app.get("/shap/{infer_run_id}")
def get_shap(infer_run_id: str, n_background: int = 50, n_explain: int = 100):
    """
    Compute SHAP values for an inference run using KernelExplainer.
    Loads the source training run's model + embeddings, builds the pair
    matrix from the inference results CSV, then returns per-dimension
    mean |SHAP| values grouped into interpretable feature groups.
    """
    import pickle
    import numpy as np
    import pandas as pd
    import torch

    # ── locate inference job ──────────────────────────────────────────────
    db = SessionLocal()
    try:
        infer_job = db.query(Job).filter(Job.run_id == infer_run_id).first()
        if not infer_job:
            return JSONResponse({"error": "inference run not found"}, status_code=404)
        source_run_id = infer_job.source_run_id
        if not source_run_id:
            return JSONResponse({"error": "no source training run linked"}, status_code=400)
        src_job = db.query(Job).filter(Job.run_id == source_run_id).first()
        src_hp  = json.loads(src_job.hyperparams) if src_job and src_job.hyperparams else {}
    finally:
        db.close()

    esm_dim      = int(src_hp.get("esm_dim", 480))
    layer_configs = src_hp.get("layer_configs", [
        {"type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3},
        {"type": "linear", "hidden_dim": 64,  "activation": "relu", "dropout": 0.2},
    ])

    # ── load model ────────────────────────────────────────────────────────
    src_dir    = os.path.join(MODELS_DIR, source_run_id)
    model_path = os.path.join(src_dir, f"model_{source_run_id}.pt")
    embed_path = os.path.join(src_dir, f"embedding_{source_run_id}.pkl")

    if not os.path.exists(model_path):
        return JSONResponse({"error": "model checkpoint not found"}, status_code=404)
    if not os.path.exists(embed_path):
        return JSONResponse({"error": "embedding file not found"}, status_code=404)

    from model_build.ppi_classifier import FlexiblePPIModel

    ckpt = torch.load(model_path, map_location="cpu")
    model = FlexiblePPIModel(ckpt["input_dim"], ckpt.get("layer_configs", layer_configs))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(embed_path, "rb") as f:
        emb_dict = pickle.load(f)

    # ── build pair matrix from inference results CSV ──────────────────────
    results_csv = os.path.join(MODELS_DIR, infer_run_id, f"results_{infer_run_id}.csv")
    if not os.path.exists(results_csv):
        return JSONResponse({"error": "results CSV not found"}, status_code=404)

    df = pd.read_csv(results_csv).dropna(subset=["probability"])
    rows = []
    for _, row in df.iterrows():
        eA = emb_dict.get(str(row["proteinA"]).strip().upper())
        eB = emb_dict.get(str(row["proteinB"]).strip().upper())
        if eA is not None and eB is not None:
            rows.append(torch.cat([eA, eB], dim=-1).float().numpy())
        if len(rows) >= n_explain + n_background:
            break

    if len(rows) < 4:
        return JSONResponse({"error": "too few embeddable pairs for SHAP"}, status_code=400)

    X = np.stack(rows)  # (N, 2*esm_dim)

    # ── KernelExplainer (model-agnostic) ─────────────────────────────────
    try:
        import shap

        def _predict(x_np):
            with torch.no_grad():
                t = torch.tensor(x_np, dtype=torch.float)
                return torch.sigmoid(model(t)).numpy()

        n_bg  = min(n_background, len(X) // 2)
        n_exp = min(n_explain, len(X) - n_bg)
        background = X[:n_bg]
        explain_X  = X[n_bg: n_bg + n_exp]

        explainer   = shap.KernelExplainer(_predict, background)
        shap_values = explainer.shap_values(explain_X, nsamples=128, silent=True)
        # shap_values: (n_exp, 2*esm_dim)
        mean_abs    = np.abs(shap_values).mean(axis=0)   # (2*esm_dim,)

    except ImportError:
        # shap not installed — fall back to gradient-free sensitivity
        baseline = X.mean(axis=0, keepdims=True)  # (1, D)
        mean_abs = np.zeros(X.shape[1])
        for d in range(X.shape[1]):
            perturbed      = baseline.copy()
            perturbed[0, d] = baseline[0, d] + baseline[0, d].std() + 1e-6
            with torch.no_grad():
                p_orig = torch.sigmoid(model(torch.tensor(baseline, dtype=torch.float))).item()
                p_pert = torch.sigmoid(model(torch.tensor(perturbed, dtype=torch.float))).item()
            mean_abs[d] = abs(p_pert - p_orig)

    # ── aggregate into feature groups ─────────────────────────────────────
    eA_shap = mean_abs[:esm_dim]
    eB_shap = mean_abs[esm_dim:]

    def _top(arr, k=15):
        idx = np.argsort(arr)[::-1][:k]
        return [{"dim": int(i), "value": float(round(arr[i], 6))} for i in idx]

    groups = {
        "eA_mean":     float(round(eA_shap.mean(), 6)),
        "eB_mean":     float(round(eB_shap.mean(), 6)),
        "eA_top":      _top(eA_shap),
        "eB_top":      _top(eB_shap),
        "global_top":  _top(mean_abs),
        "all_dims":    mean_abs.tolist(),
        "esm_dim":     esm_dim,
    }
    return groups



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
