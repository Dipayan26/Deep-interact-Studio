import hashlib
import json
import math
import os
import pickle
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from collections import defaultdict, deque
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


def _queue_unavailable_response(run_id: str) -> "JSONResponse":
    return JSONResponse(
        {
            "error": "Job queue is unavailable. Please try again later.",
            "run_id": run_id,
        },
        status_code=503,
    )

from fastapi import Body, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import text
from typing import List

from database import Base, engine, SessionLocal
from models import Job
from tasks import (
    train_ppi_model, run_ppi_inference,
    train_dtpi_model, run_dtpi_inference_task,
    train_rpi_model, run_rpi_inference_task,
    train_pdi_model, run_pdi_inference_task,
)
from model_build.chemberta_embed import (
    compute_and_save_chem_embeddings,
    compute_and_save_chunked_chem_embeddings,
)
from model_build.dnabert_embed import (
    compute_and_save_dna_embeddings,
    compute_and_save_chunked_dna_embeddings,
)
from model_build.dtpi_infer import run_dtpi_inference as score_dtpi_inference
from model_build.esm_embed import (
    compute_and_save_chunked_embeddings,
    compute_and_save_embeddings,
)
from model_build.pdi_infer import run_pdi_inference as score_pdi_inference
from model_build.ppi_infer import run_inference as score_ppi_inference
from model_build.rnafm_embed import (
    compute_and_save_chunked_rna_embeddings,
    compute_and_save_rna_embeddings,
)
from model_build.rpi_infer import run_rpi_inference as score_rpi_inference

MODELS_DIR = "/app/saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)
EXAMPLE_RUNS_DIR = os.getenv(
    "EXAMPLE_RUNS_DIR",
    os.path.join(os.path.dirname(__file__), "example_runs"),
)
EXAMPLE_INFERENCE_RUNS_DIR = os.getenv(
    "EXAMPLE_INFERENCE_RUNS_DIR",
    os.path.join(os.path.dirname(__file__), "example_inference_runs"),
)

EXAMPLE_RUNS = {
    "example_ppi": {
        "task_type": "ppi",
        "title": "Example PPI Model",
        "available": True,
        "directory": "ppi",
    },
    "example_dtpi": {
        "task_type": "dtpi",
        "title": "Example DTPI Model",
        "available": True,
        "directory": "dtpi",
    },
    "example_rpi": {
        "task_type": "rpi",
        "title": "Example RPI Model",
        "available": True,
        "directory": "rpi",
    },
    "example_pdi": {
        "task_type": "pdi",
        "title": "Example PDI Model",
        "available": False,
        "directory": "pdi",
    },
}

EXAMPLE_INFERENCE_RUNS = {
    "example_infer_ppi": {
        "task_type": "ppi",
        "title": "Example PPI Inference",
        "available": True,
        "directory": "ppi",
    },
    "example_infer_dtpi": {
        "task_type": "dtpi",
        "title": "Example DTPI Inference",
        "available": True,
        "directory": "dtpi",
    },
    "example_infer_rpi": {
        "task_type": "rpi",
        "title": "Example RPI Inference",
        "available": True,
        "directory": "rpi",
    },
    "example_infer_pdi": {
        "task_type": "pdi",
        "title": "Example PDI Inference",
        "available": False,
        "directory": "pdi",
    },
}


def _example_run_info(example_id: str) -> dict | None:
    info = EXAMPLE_RUNS.get(example_id)
    if not info or not info.get("available"):
        return None
    return info


def _example_run_dir(example_id: str) -> str | None:
    info = _example_run_info(example_id)
    if not info:
        return None
    return os.path.join(EXAMPLE_RUNS_DIR, info["directory"])


def _read_example_json(example_id: str, filename: str):
    run_dir = _example_run_dir(example_id)
    if not run_dir:
        return JSONResponse({"error": "example run not found"}, status_code=404)
    path = os.path.join(run_dir, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": f"{filename} not found"}, status_code=404)
    with open(path) as f:
        return json.load(f)


def _example_inference_info(example_id: str) -> dict | None:
    info = EXAMPLE_INFERENCE_RUNS.get(example_id)
    if not info or not info.get("available"):
        return None
    return info


def _example_inference_dir(example_id: str) -> str | None:
    info = _example_inference_info(example_id)
    if not info:
        return None
    return os.path.join(EXAMPLE_INFERENCE_RUNS_DIR, info["directory"])


def _read_example_inference_json(example_id: str, filename: str):
    run_dir = _example_inference_dir(example_id)
    if not run_dir:
        return JSONResponse({"error": "example inference run not found"}, status_code=404)
    path = os.path.join(run_dir, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": f"{filename} not found"}, status_code=404)
    with open(path) as f:
        return json.load(f)


def _int_env(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(min_value, value)


MAX_UPLOAD_FILE_MB    = _int_env("MAX_UPLOAD_FILE_MB", 100)
MAX_UPLOAD_REQUEST_MB = _int_env("MAX_UPLOAD_REQUEST_MB", 100)
MAX_UPLOAD_FILE_BYTES = MAX_UPLOAD_FILE_MB * 1024 * 1024
MAX_UPLOAD_REQUEST_BYTES = MAX_UPLOAD_REQUEST_MB * 1024 * 1024

RATE_LIMIT_RUN_INFERENCE_PER_MIN = _int_env("RATE_LIMIT_RUN_INFERENCE_PER_MIN", 20)
TRAINING_JOB_QUOTA_PER_IP = _int_env("TRAINING_JOB_QUOTA_PER_IP", 10)
TRAINING_JOB_QUOTA_WINDOW_HOURS = _int_env("TRAINING_JOB_QUOTA_WINDOW_HOURS", 3)
MAX_ACTIVE_TRAINING_JOBS = _int_env("MAX_ACTIVE_TRAINING_JOBS", 20)
BATCH_INFERENCE_QUOTA_PER_IP = _int_env("BATCH_INFERENCE_QUOTA_PER_IP", 15)
SINGLE_INFERENCE_QUOTA_PER_IP = _int_env("SINGLE_INFERENCE_QUOTA_PER_IP", 30)
INFERENCE_QUOTA_WINDOW_HOURS = _int_env("INFERENCE_QUOTA_WINDOW_HOURS", 5)
RATE_LIMIT_WINDOW_SECONDS = 60
_UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_MODEL_PARAMS = 5_000_000
TRANSFORMER_MAX_POSITIONS = 4096
MAX_SINGLE_PAIR_INPUT_LEN = 512


class UploadTooLargeError(Exception):
    pass


class _InMemoryRateLimiter:
    def __init__(self):
        self._buckets = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            q = self._buckets[key]
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= limit:
                retry_after = max(1, int((q[0] + window_seconds) - now + 0.999))
                return False, retry_after
            q.append(now)
            return True, 0


_rate_limiter = _InMemoryRateLimiter()


class _RollingQuotaLimiter:
    def __init__(self):
        self._buckets = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int, int]:
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            q = self._buckets[key]
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= limit:
                retry_after = max(1, int((q[0] + window_seconds) - now + 0.999))
                return False, 0, retry_after
            q.append(now)
            return True, limit - len(q), 0


_training_quota_limiter = _RollingQuotaLimiter()
_inference_quota_limiter = _RollingQuotaLimiter()


def _is_upload_endpoint(method: str, path: str) -> bool:
    if method != "POST":
        return False
    return path == "/create_job" or path.startswith("/run_inference/")


def _rate_limit_for(method: str, path: str) -> tuple[int, str] | None:
    if method != "POST":
        return None
    if path.startswith("/run_inference/"):
        return RATE_LIMIT_RUN_INFERENCE_PER_MIN, "/run_inference/{source_run_id}"
    if path.startswith("/run_single_inference/"):
        return RATE_LIMIT_RUN_INFERENCE_PER_MIN, "/run_single_inference/{source_run_id}"
    return None


def _client_identifier(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip() if forwarded else ""
    if first:
        return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _active_training_job_count() -> int:
    db = SessionLocal()
    try:
        return (
            db.query(Job)
            .filter(Job.job_type == "train", Job.status.in_(["queued", "running"]))
            .count()
        )
    finally:
        db.close()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def enforce_request_limits(request: Request, call_next):
    method = request.method.upper()
    path = request.url.path

    rl_cfg = _rate_limit_for(method, path)
    if rl_cfg:
        limit, route_key = rl_cfg
        client_key = f"{_client_identifier(request)}:{route_key}"
        allowed, retry_after = _rate_limiter.allow(
            client_key, limit=limit, window_seconds=RATE_LIMIT_WINDOW_SECONDS
        )
        if not allowed:
            return JSONResponse(
                {
                    "error": (
                        f"Rate limit exceeded for {route_key}. "
                        f"Allowed: {limit} requests per {RATE_LIMIT_WINDOW_SECONDS} seconds."
                    )
                },
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

    if method == "POST" and path == "/create_job":
        active_training_jobs = _active_training_job_count()
        if active_training_jobs >= MAX_ACTIVE_TRAINING_JOBS:
            return JSONResponse(
                {
                    "error": (
                        "Training queue is full. "
                        f"Maximum {MAX_ACTIVE_TRAINING_JOBS} queued or running training jobs "
                        "are allowed platform-wide."
                    )
                },
                status_code=429,
                headers={
                    "Retry-After": "600",
                    "X-Queue-Limit": str(MAX_ACTIVE_TRAINING_JOBS),
                    "X-Active-Training-Jobs": str(active_training_jobs),
                },
            )

        client_id = _client_identifier(request)
        quota_window_seconds = TRAINING_JOB_QUOTA_WINDOW_HOURS * 3600
        allowed, remaining, retry_after = _training_quota_limiter.allow(
            f"{client_id}:training",
            TRAINING_JOB_QUOTA_PER_IP,
            quota_window_seconds,
        )
        if not allowed:
            return JSONResponse(
                {
                    "error": (
                        "Training job quota exceeded. "
                        f"Maximum {TRAINING_JOB_QUOTA_PER_IP} training jobs per IP "
                        f"per {TRAINING_JOB_QUOTA_WINDOW_HOURS} hours."
                    )
                },
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-Quota-Limit": str(TRAINING_JOB_QUOTA_PER_IP),
                    "X-Quota-Window-Hours": str(TRAINING_JOB_QUOTA_WINDOW_HOURS),
                },
            )

    quota_window_seconds = INFERENCE_QUOTA_WINDOW_HOURS * 3600
    if method == "POST" and path.startswith("/run_inference/"):
        client_id = _client_identifier(request)
        allowed, remaining, retry_after = _inference_quota_limiter.allow(
            f"{client_id}:batch",
            BATCH_INFERENCE_QUOTA_PER_IP,
            quota_window_seconds,
        )
        if not allowed:
            return JSONResponse(
                {
                    "error": (
                        "Batch inference quota exceeded. "
                        f"Maximum {BATCH_INFERENCE_QUOTA_PER_IP} batch inference jobs per IP "
                        f"per {INFERENCE_QUOTA_WINDOW_HOURS} hours."
                    )
                },
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-Quota-Limit": str(BATCH_INFERENCE_QUOTA_PER_IP),
                    "X-Quota-Window-Hours": str(INFERENCE_QUOTA_WINDOW_HOURS),
                },
            )

    if method == "POST" and path.startswith("/run_single_inference/"):
        client_id = _client_identifier(request)
        allowed, remaining, retry_after = _inference_quota_limiter.allow(
            f"{client_id}:single",
            SINGLE_INFERENCE_QUOTA_PER_IP,
            quota_window_seconds,
        )
        if not allowed:
            return JSONResponse(
                {
                    "error": (
                        "Single-pair inference quota exceeded. "
                        f"Maximum {SINGLE_INFERENCE_QUOTA_PER_IP} single-pair predictions per IP "
                        f"per {INFERENCE_QUOTA_WINDOW_HOURS} hours."
                    )
                },
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-Quota-Limit": str(SINGLE_INFERENCE_QUOTA_PER_IP),
                    "X-Quota-Window-Hours": str(INFERENCE_QUOTA_WINDOW_HOURS),
                },
            )

    if _is_upload_endpoint(method, path):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                req_size = int(content_length)
            except ValueError:
                return JSONResponse({"error": "Invalid Content-Length header."}, status_code=400)
            if req_size > MAX_UPLOAD_REQUEST_BYTES:
                return JSONResponse(
                    {
                        "error": (
                            f"Request too large. Maximum allowed request size is "
                            f"{MAX_UPLOAD_REQUEST_MB} MB."
                        )
                    },
                    status_code=413,
                )

    return await call_next(request)


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
    _start_cleanup_scheduler()


def _start_cleanup_scheduler():
    def _loop():
        while True:
            time.sleep(3600)  # run every hour
            try:
                _run_cleanup()
            except Exception as exc:
                print(f"[cleanup-scheduler] error: {exc}", flush=True)

    t = threading.Thread(target=_loop, daemon=True, name="cleanup-scheduler")
    t.start()
    print("[cleanup-scheduler] started (hourly)", flush=True)


def _run_cleanup(failed_ttl_days: int = 1, completed_ttl_days: int = 30):
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
        f"[cleanup] sweep — stale→failed: {marked_stale}, "
        f"dirs removed: {deleted_dirs}, "
        f"DB rows purged (failed/cancelled>{failed_ttl_days}d / completed>{completed_ttl_days}d): {deleted_rows}",
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
    if not files:
        raise ValueError("No files uploaded.")

    total_size = 0
    paths = []
    for file in files:
        safe_name = os.path.basename(file.filename or "upload.csv")
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", safe_name) or "upload.csv"
        dest = os.path.join(run_dir, f"{run_id}_{safe_name}")
        file_size = 0
        with open(dest, "wb") as f:
            while True:
                chunk = file.file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break

                chunk_len = len(chunk)
                file_size += chunk_len
                total_size += chunk_len

                if file_size > MAX_UPLOAD_FILE_BYTES:
                    raise UploadTooLargeError(
                        f"File '{safe_name}' exceeds the per-file limit of {MAX_UPLOAD_FILE_MB} MB."
                    )
                if total_size > MAX_UPLOAD_REQUEST_BYTES:
                    raise UploadTooLargeError(
                        f"Upload exceeds the total request limit of {MAX_UPLOAD_REQUEST_MB} MB."
                    )
                f.write(chunk)
        paths.append(dest)
    return paths


def _model_input_dim(task_type: str, hp: dict) -> int:
    if "input_dim" in hp:
        return int(hp["input_dim"])
    representation_mode = str(
        hp.get("embedding_representation", hp.get("representation_mode", "pooled"))
    ).lower()
    if representation_mode == "chunked":
        if task_type == "dtpi":
            return int(hp.get("chunk_model_dim", max(int(hp.get("chem_dim", 768)), int(hp.get("esm_dim", 480)))))
        if task_type == "rpi":
            return int(hp.get("chunk_model_dim", max(int(hp.get("rna_dim", 640)), int(hp.get("esm_dim", 480)))))
        if task_type == "pdi":
            return int(hp.get("chunk_model_dim", max(int(hp.get("dna_dim", 768)), int(hp.get("esm_dim", 480)))))
        return int(hp.get("esm_dim", 480))
    if task_type == "dtpi":
        return int(hp.get("chem_dim", 768)) + int(hp.get("esm_dim", 480))
    if task_type == "rpi":
        return int(hp.get("rna_dim", 640)) + int(hp.get("esm_dim", 480))
    if task_type == "pdi":
        return int(hp.get("dna_dim", 768)) + int(hp.get("esm_dim", 480))
    return 2 * int(hp.get("esm_dim", 480))


def _estimated_model_params(input_dim: int, layer_configs: list, sequence_mode: bool = False) -> int:
    total = 0
    cur = input_dim
    for cfg in layer_configs:
        lt = str(cfg.get("type", "")).lower()
        if lt == "linear":
            h = int(cfg.get("hidden_dim", 256))
            total += cur * h + h
            if cfg.get("batchnorm"):
                total += 2 * h
            cur = h
        elif lt == "cnn1d":
            out_ch = int(cfg.get("out_channels", 64))
            k = int(cfg.get("kernel_size", 3))
            in_ch = cur if sequence_mode else 1
            total += in_ch * out_ch * k + out_ch
            cur = out_ch
        elif lt == "bilstm":
            h = int(cfg.get("hidden_size", 128))
            nl = int(cfg.get("num_layers", 1))
            gate = 4
            dirs = 2
            total += dirs * gate * (cur * h + h * h + 2 * h)
            for _ in range(nl - 1):
                total += dirs * gate * (dirs * h * h + h * h + 2 * h)
            cur = dirs * h
        elif lt == "gru":
            h = int(cfg.get("hidden_size", 128))
            nl = int(cfg.get("num_layers", 1))
            bidir = bool(cfg.get("bidirectional", True))
            gate = 3
            dirs = 2 if bidir else 1
            total += dirs * gate * (cur * h + h * h + 2 * h)
            for _ in range(nl - 1):
                total += dirs * gate * (dirs * h * h + h * h + 2 * h)
            cur = dirs * h
        elif lt == "transformer":
            d = int(cfg.get("d_model", 256))
            ff = int(cfg.get("dim_feedforward", d * 2))
            nl = int(cfg.get("num_layers", 2))
            total += cur * d + d
            if sequence_mode:
                total += TRANSFORMER_MAX_POSITIONS * d
            total += nl * (4 * d * d + 4 * d + d * ff + ff + ff * d + d + 4 * d)
            cur = d
        elif lt == "residual":
            h = int(cfg.get("hidden_dim", 256))
            total += cur * h + h + h * cur + cur
            if cfg.get("batchnorm"):
                total += 2 * h
            total += 2 * cur
    head_in = 2 * cur if sequence_mode else cur
    total += head_in + 1
    return total


def _param_limit_error(task_type: str, hp: dict) -> str | None:
    layer_configs = hp.get("layer_configs", [])
    if not isinstance(layer_configs, list) or not layer_configs:
        return None

    input_dim = _model_input_dim(task_type, hp)
    representation_mode = str(
        hp.get("embedding_representation", hp.get("representation_mode", "pooled"))
    ).lower()
    sequence_mode = representation_mode == "chunked"
    n_params = _estimated_model_params(input_dim, layer_configs, sequence_mode=sequence_mode)
    if sequence_mode and task_type in {"dtpi", "rpi", "pdi"}:
        left_dim_key = {"dtpi": "chem_dim", "rpi": "rna_dim", "pdi": "dna_dim"}[task_type]
        left_dim = int(hp.get(left_dim_key, 768 if task_type != "rpi" else 640))
        right_dim = int(hp.get("esm_dim", 480))
        n_params += left_dim * input_dim + input_dim
        n_params += right_dim * input_dim + input_dim
        n_params += 2 * input_dim
    if n_params > MAX_MODEL_PARAMS:
        return (
            f"Model has {n_params:,} estimated parameters; "
            f"maximum allowed is {MAX_MODEL_PARAMS:,}."
        )
    return None


def _leakage_warnings(task_type: str, csv_paths: list[str]) -> list[str]:
    """
    Lightweight leakage-risk checks on uploaded training data.
    Returns human-readable warnings; never raises for bad input.
    """
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split

        if not csv_paths:
            return []

        df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
        if df.empty:
            return []

        warnings: list[str] = []
        if "label" in df.columns:
            labels = df["label"].astype(float).astype(int).tolist()
        else:
            labels = [0] * len(df)

        dup_rows = int(df.duplicated().sum())
        if dup_rows > 0:
            warnings.append(
                f"Detected {dup_rows} exact duplicate row(s). Duplicates can leak signals into validation."
            )

        idx = list(range(len(df)))
        try:
            tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)
        except Exception:
            tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=42)

        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[va_idx].reset_index(drop=True)

        def _ratio(a: set, b: set) -> float:
            return (len(a & b) / max(1, len(a | b)))

        if task_type == "ppi" and {"proteinA", "proteinB"}.issubset(df.columns):
            def _norm_seq(s):
                return str(s).strip().upper()

            p_a = df["proteinA"].map(_norm_seq)
            p_b = df["proteinB"].map(_norm_seq)

            pair_key_all = p_a.combine(p_b, lambda a, b: "|".join(sorted((a, b))))
            rev_dups = int(pair_key_all.duplicated().sum())
            if rev_dups > 0:
                warnings.append(
                    f"Detected {rev_dups} duplicate/reverse PPI pair(s) (A,B)/(B,A). "
                    "These can cause train/validation leakage."
                )

            tr_keys = set(
                tr_df["proteinA"].map(_norm_seq).combine(
                    tr_df["proteinB"].map(_norm_seq), lambda a, b: "|".join(sorted((a, b)))
                )
            )
            va_keys = set(
                va_df["proteinA"].map(_norm_seq).combine(
                    va_df["proteinB"].map(_norm_seq), lambda a, b: "|".join(sorted((a, b)))
                )
            )
            pair_overlap = len(tr_keys & va_keys)
            if pair_overlap > 0:
                warnings.append(
                    f"{pair_overlap} pair key(s) appear in both train and validation under random split. "
                    "Deduplicate pairs before training."
                )

            tr_entities = set(tr_df["proteinA"].map(_norm_seq)) | set(tr_df["proteinB"].map(_norm_seq))
            va_entities = set(va_df["proteinA"].map(_norm_seq)) | set(va_df["proteinB"].map(_norm_seq))
            ent_overlap = _ratio(tr_entities, va_entities)
            if ent_overlap >= 0.30:
                warnings.append(
                    f"High protein overlap between train/validation (~{ent_overlap*100:.1f}% Jaccard). "
                    "For stricter generalization, consider protein-disjoint splitting."
                )

        elif task_type == "dtpi" and {"smiles", "sequence"}.issubset(df.columns):
            def _norm_smiles(s):
                return str(s).strip()
            def _norm_seq(s):
                return str(s).strip().upper()

            pair_key_all = df["smiles"].map(_norm_smiles) + "|" + df["sequence"].map(_norm_seq)
            pair_dups = int(pair_key_all.duplicated().sum())
            if pair_dups > 0:
                warnings.append(
                    f"Detected {pair_dups} duplicate DTPI pair row(s). Duplicates can leak into validation."
                )

            tr_smiles = set(tr_df["smiles"].map(_norm_smiles))
            va_smiles = set(va_df["smiles"].map(_norm_smiles))
            tr_prots  = set(tr_df["sequence"].map(_norm_seq))
            va_prots  = set(va_df["sequence"].map(_norm_seq))
            sm_overlap = _ratio(tr_smiles, va_smiles)
            pr_overlap = _ratio(tr_prots, va_prots)
            if sm_overlap >= 0.30 or pr_overlap >= 0.30:
                warnings.append(
                    f"High entity overlap (SMILES ~{sm_overlap*100:.1f}%, proteins ~{pr_overlap*100:.1f}% Jaccard). "
                    "Consider scaffold/protein-disjoint split for stricter evaluation."
                )

        elif task_type == "rpi" and {"rna_sequence", "protein_sequence"}.issubset(df.columns):
            def _norm_rna(s):
                return str(s).strip().upper().replace("T", "U")
            def _norm_seq(s):
                return str(s).strip().upper()

            pair_key_all = df["rna_sequence"].map(_norm_rna) + "|" + df["protein_sequence"].map(_norm_seq)
            pair_dups = int(pair_key_all.duplicated().sum())
            if pair_dups > 0:
                warnings.append(
                    f"Detected {pair_dups} duplicate RPI pair row(s). Duplicates can leak into validation."
                )

            tr_rna = set(tr_df["rna_sequence"].map(_norm_rna))
            va_rna = set(va_df["rna_sequence"].map(_norm_rna))
            tr_prot = set(tr_df["protein_sequence"].map(_norm_seq))
            va_prot = set(va_df["protein_sequence"].map(_norm_seq))
            rna_overlap = _ratio(tr_rna, va_rna)
            prot_overlap = _ratio(tr_prot, va_prot)
            if rna_overlap >= 0.30 or prot_overlap >= 0.30:
                warnings.append(
                    f"High entity overlap (RNA ~{rna_overlap*100:.1f}%, proteins ~{prot_overlap*100:.1f}% Jaccard). "
                    "Consider RNA/protein-disjoint split for stricter evaluation."
                )

        elif task_type == "pdi" and {"dna_sequence", "protein_sequence"}.issubset(df.columns):
            def _norm_dna(s):
                return str(s).strip().upper()
            def _norm_seq(s):
                return str(s).strip().upper()

            pair_key_all = df["dna_sequence"].map(_norm_dna) + "|" + df["protein_sequence"].map(_norm_seq)
            pair_dups = int(pair_key_all.duplicated().sum())
            if pair_dups > 0:
                warnings.append(
                    f"Detected {pair_dups} duplicate PDI pair row(s). Duplicates can leak into validation."
                )

            tr_dna = set(tr_df["dna_sequence"].map(_norm_dna))
            va_dna = set(va_df["dna_sequence"].map(_norm_dna))
            tr_prot = set(tr_df["protein_sequence"].map(_norm_seq))
            va_prot = set(va_df["protein_sequence"].map(_norm_seq))
            dna_overlap = _ratio(tr_dna, va_dna)
            prot_overlap = _ratio(tr_prot, va_prot)
            if dna_overlap >= 0.30 or prot_overlap >= 0.30:
                warnings.append(
                    f"High entity overlap (DNA ~{dna_overlap*100:.1f}%, proteins ~{prot_overlap*100:.1f}% Jaccard). "
                    "Consider DNA/protein-disjoint split for stricter evaluation."
                )

        return warnings
    except Exception as exc:
        return [f"Leakage check could not be completed: {exc}"]


def _job_bundle_metadata(job: Job) -> dict:
    hp = {}
    if job.hyperparams:
        try:
            hp = json.loads(job.hyperparams)
        except Exception:
            hp = {}

    metrics = {}
    if job.metrics:
        try:
            metrics = json.loads(job.metrics)
        except Exception:
            metrics = {}

    return {
        "run_id": job.run_id,
        "status": job.status,
        "job_type": job.job_type,
        "source_run_id": job.source_run_id,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "hyperparams": hp,
        "metrics": metrics,
        "result": job.result,
    }


def _bundle_candidates(run_id: str, run_dir: str) -> list[str]:
    names = []

    # Core known artifacts first.
    core = [
        f"model_{run_id}.pt",
        f"embedding_{run_id}.pkl",
        f"chem_embedding_{run_id}.pkl",
        f"rna_embedding_{run_id}.pkl",
        f"dna_embedding_{run_id}.pkl",
        f"metrics_{run_id}.json",
        f"infer_metrics_{run_id}.json",
        f"results_{run_id}.csv",
    ]
    for fn in core:
        full = os.path.join(run_dir, fn)
        if os.path.isfile(full):
            names.append(fn)

    # Include uploaded input files and any other embedding caches for this run.
    for fn in sorted(os.listdir(run_dir)):
        if fn in names:
            continue
        if fn == f"artifacts_{run_id}.zip":
            continue
        if fn.startswith(f"{run_id}_") and fn.endswith(".csv"):
            names.append(fn)
            continue
        if fn.endswith(".pkl") and ("embedding" in fn or fn.startswith("new_")):
            names.append(fn)
            continue
        if fn.startswith(("metrics_", "infer_metrics_")) and fn.endswith(".json"):
            names.append(fn)
            continue

    return names


def _build_artifact_bundle(run_id: str, job: Job) -> tuple[str, int]:
    run_dir = os.path.join(MODELS_DIR, run_id)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError("artifact directory not found")

    zip_name = f"artifacts_{run_id}.zip"
    zip_path = os.path.join(run_dir, zip_name)
    members = _bundle_candidates(run_id, run_dir)
    if not members:
        raise FileNotFoundError("no artifacts found for this run")

    metadata = _job_bundle_metadata(job)

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"config_{run_id}.json",
            json.dumps(metadata, indent=2, ensure_ascii=True),
        )
        for fn in members:
            zf.write(os.path.join(run_dir, fn), arcname=fn)

    return zip_path, len(members) + 1


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """Return aggregate platform usage counts."""
    db = SessionLocal()
    try:
        total_train     = db.query(Job).filter(Job.job_type == "train").count()
        total_inference = db.query(Job).filter(Job.job_type == "inference").count()
        completed       = db.query(Job).filter(Job.job_type == "train", Job.status == "completed").count()
        return {
            "total_training_jobs":   total_train,
            "total_inference_jobs":  total_inference,
            "completed_training":    completed,
        }
    finally:
        db.close()



# ---------------------------------------------------------------------------
# POST /create_job  — queue a training job
# ---------------------------------------------------------------------------

@app.post("/create_job")
async def create_job(
    files:       List[UploadFile] = File(...),
    hyperparams: str              = Form("{}"),   # JSON string
):
    run_id  = str(uuid.uuid4())[:8]

    # validate hyperparams JSON
    try:
        hp = json.loads(hyperparams)
    except Exception:
        hp = {}
    task_type = hp.get("task_type", "ppi")
    try:
        param_error = _param_limit_error(task_type, hp)
    except Exception:
        param_error = "Could not validate model parameter count."
    if param_error:
        return JSONResponse({"error": param_error}, status_code=400)

    run_dir = _run_dir(run_id)
    try:
        paths = _save_uploaded_files(files, run_dir, run_id)
    except UploadTooLargeError as e:
        shutil.rmtree(run_dir, ignore_errors=True)
        return JSONResponse({"error": str(e)}, status_code=413)
    except ValueError as e:
        shutil.rmtree(run_dir, ignore_errors=True)
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        return JSONResponse({"error": "Could not process uploaded files."}, status_code=500)

    leakage_warnings = _leakage_warnings(task_type=task_type, csv_paths=paths)
    if leakage_warnings:
        hp["leakage_warnings"] = leakage_warnings

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

    try:
        if task_type == "dtpi":
            task = train_dtpi_model.delay(run_id, paths, json.dumps(hp))
        elif task_type == "rpi":
            task = train_rpi_model.delay(run_id, paths, json.dumps(hp))
        elif task_type == "pdi":
            task = train_pdi_model.delay(run_id, paths, json.dumps(hp))
        else:
            task = train_ppi_model.delay(run_id, paths, json.dumps(hp))
    except Exception as exc:
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.run_id == run_id).first()
            if job:
                job.status = "failed"
                job.result = f"Queue submission failed: {exc}"
                db.commit()
        finally:
            db.close()
        return _queue_unavailable_response(run_id)

    # store celery task id so we can revoke it later
    db  = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        job.celery_task_id = task.id
        db.commit()
    finally:
        db.close()

    return {"run_id": run_id, "cancel_token": cancel_token, "leakage_warnings": leakage_warnings}


# ---------------------------------------------------------------------------
# GET /check_status/{run_id}
# ---------------------------------------------------------------------------

@app.get("/example_runs")
def list_example_runs():
    runs = []
    for example_id, info in EXAMPLE_RUNS.items():
        runs.append({
            "example_id": example_id,
            "task_type": info["task_type"],
            "title": info["title"],
            "available": bool(info.get("available")),
        })
    return {"examples": runs}


@app.get("/example_runs/{example_id}/check_status")
def example_check_status(example_id: str):
    metadata = _read_example_json(example_id, "metadata.json")
    if isinstance(metadata, JSONResponse):
        return metadata

    return {
        "run_id": metadata.get("run_id", example_id),
        "status": metadata.get("status", "completed"),
        "job_type": metadata.get("job_type", "train"),
        "result": metadata.get("result"),
        "source_run_id": metadata.get("source_run_id"),
        "title": metadata.get("title"),
        "is_example": True,
        "hyperparams": metadata.get("hyperparams", {}),
    }


@app.get("/example_runs/{example_id}/metrics")
def example_metrics(example_id: str):
    return _read_example_json(example_id, "metrics.json")


@app.get("/example_runs/{example_id}/dataset_stats")
def example_dataset_stats(example_id: str):
    return _read_example_json(example_id, "dataset_stats.json")


@app.get("/example_runs/{example_id}/umap_data")
def example_umap_data(example_id: str):
    return _read_example_json(example_id, "emb_umap.json")


@app.get("/example_runs/{example_id}/model_umap")
def example_model_umap(example_id: str):
    return _read_example_json(example_id, "model_umap.json")


@app.get("/example_inference_runs")
def list_example_inference_runs():
    runs = []
    for example_id, info in EXAMPLE_INFERENCE_RUNS.items():
        runs.append({
            "example_id": example_id,
            "task_type": info["task_type"],
            "title": info["title"],
            "available": bool(info.get("available")),
        })
    return {"examples": runs}


@app.get("/example_inference_runs/{example_id}/check_status")
def example_inference_check_status(example_id: str):
    metadata = _read_example_inference_json(example_id, "metadata.json")
    if isinstance(metadata, JSONResponse):
        return metadata
    return {
        "run_id": metadata.get("run_id", example_id),
        "status": metadata.get("status", "completed"),
        "job_type": metadata.get("job_type", "inference"),
        "result": metadata.get("result"),
        "source_run_id": metadata.get("source_run_id"),
        "title": metadata.get("title"),
        "is_example": True,
        "hyperparams": metadata.get("hyperparams", {}),
    }


@app.get("/example_inference_runs/{example_id}/job_detail")
def example_inference_job_detail(example_id: str):
    metadata = _read_example_inference_json(example_id, "metadata.json")
    if isinstance(metadata, JSONResponse):
        return metadata
    return {
        "run_id": metadata.get("run_id", example_id),
        "status": metadata.get("status", "completed"),
        "job_type": metadata.get("job_type", "inference"),
        "hyperparams": metadata.get("hyperparams", {}),
        "source_run_id": metadata.get("source_run_id"),
        "task_type": metadata.get("task_type"),
        "title": metadata.get("title"),
        "is_example": True,
    }


@app.get("/example_inference_runs/{example_id}/source_job_detail")
def example_inference_source_job_detail(example_id: str):
    return _read_example_inference_json(example_id, "source_metadata.json")


@app.get("/example_inference_runs/{example_id}/source_metrics")
def example_inference_source_metrics(example_id: str):
    return _read_example_inference_json(example_id, "source_metrics.json")


@app.get("/example_inference_runs/{example_id}/download_results")
def example_inference_download_results(example_id: str):
    run_dir = _example_inference_dir(example_id)
    if not run_dir:
        return JSONResponse({"error": "example inference run not found"}, status_code=404)
    metadata = _read_example_inference_json(example_id, "metadata.json")
    if isinstance(metadata, JSONResponse):
        return metadata
    path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(path):
        return JSONResponse({"error": "results not found"}, status_code=404)
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"example_results_{metadata.get('task_type', example_id)}.csv",
    )


@app.get("/example_inference_runs/{example_id}/inference_metrics")
def example_inference_metrics(example_id: str):
    return _read_example_inference_json(example_id, "inference_metrics.json")


@app.get("/example_inference_runs/{example_id}/shap")
def example_inference_shap(example_id: str):
    return _read_example_inference_json(example_id, "shap.json")


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
# POST /run_single_inference/{source_run_id}  — direct single-pair inference
# ---------------------------------------------------------------------------

def _load_completed_training_source(source_run_id: str):
    if not _valid_run_id(source_run_id):
        return None, JSONResponse({"error": "invalid source_run_id"}, status_code=400)

    db = SessionLocal()
    try:
        src = db.query(Job).filter(Job.run_id == source_run_id).first()
        if not src:
            return None, JSONResponse({"error": "source run_id not found"}, status_code=404)
        if src.status != "completed":
            return None, JSONResponse({"error": "source job not completed yet"}, status_code=400)
        if src.job_type != "train":
            return None, JSONResponse({"error": "source job is not a training run"}, status_code=400)
        src_hp = json.loads(src.hyperparams) if src.hyperparams else {}
        return src_hp, None
    except Exception:
        return None, JSONResponse({"error": "Could not load source job."}, status_code=500)
    finally:
        db.close()


def _require_payload_fields(payload: dict, fields: list[str]) -> dict | None:
    missing = [f for f in fields if not str(payload.get(f, "")).strip()]
    if missing:
        return {"error": f"Missing required field(s): {', '.join(missing)}"}
    return None


def _require_max_lengths(payload: dict, fields: list[str]) -> dict | None:
    too_long = [
        f"{f} ({len(str(payload.get(f, '')))} > {MAX_SINGLE_PAIR_INPUT_LEN})"
        for f in fields
        if len(str(payload.get(f, ""))) > MAX_SINGLE_PAIR_INPUT_LEN
    ]
    if too_long:
        return {
            "error": (
                "Single-pair inputs must be at most "
                f"{MAX_SINGLE_PAIR_INPUT_LEN} characters each: {', '.join(too_long)}"
            )
        }
    return None


def _single_result_payload(task_type: str, result: dict) -> dict:
    prob = result.get("probability")
    pred = result.get("prediction")
    if prob is not None:
        prob = round(float(prob), 4)
    if pred is not None:
        pred = int(pred)
    return {
        "task_type": task_type,
        "result": {
            **result,
            "probability": prob,
            "prediction": pred,
        },
    }


@app.post("/run_single_inference/{source_run_id}")
def run_single_inference(source_run_id: str, payload: dict = Body(...)):
    src_hp, error_response = _load_completed_training_source(source_run_id)
    if error_response is not None:
        return error_response

    import pandas as pd

    task_type = src_hp.get("task_type", "ppi")
    src_dir = os.path.join(MODELS_DIR, source_run_id)
    model_path = os.path.join(src_dir, f"model_{source_run_id}.pt")
    if not os.path.exists(model_path):
        return JSONResponse({"error": "Trained model not found."}, status_code=404)

    representation_mode = str(
        src_hp.get("embedding_representation", src_hp.get("representation_mode", "pooled"))
    ).lower()
    use_chunked = representation_mode == "chunked"
    chunk_dtype = src_hp.get("chunk_dtype", "float16")

    try:
        with tempfile.TemporaryDirectory(prefix="single_infer_") as tmp_dir:
            if task_type == "dtpi":
                field_error = _require_payload_fields(payload, ["smiles", "sequence"])
                if field_error:
                    return JSONResponse(field_error, status_code=400)
                length_error = _require_max_lengths(payload, ["smiles", "sequence"])
                if length_error:
                    return JSONResponse(length_error, status_code=400)

                smiles = str(payload["smiles"]).strip()
                seq = str(payload["sequence"]).strip().upper()
                df = pd.DataFrame([{"smiles": smiles, "sequence": seq}])

                chem_embed_path = os.path.join(src_dir, f"chem_embedding_{source_run_id}.pkl")
                esm_embed_path = os.path.join(src_dir, f"embedding_{source_run_id}.pkl")
                if not os.path.exists(chem_embed_path) or not os.path.exists(esm_embed_path):
                    return JSONResponse({"error": "Required source embeddings not found."}, status_code=404)
                with open(chem_embed_path, "rb") as f:
                    chem_dict = pickle.load(f)
                with open(esm_embed_path, "rb") as f:
                    esm_dict = pickle.load(f)

                if smiles not in chem_dict:
                    tmp_chem = os.path.join(tmp_dir, "chem.pkl")
                    if use_chunked:
                        compute_and_save_chunked_chem_embeddings(
                            [smiles],
                            tmp_chem,
                            src_hp.get("chem_model", "seyonec/ChemBERTa-zinc-base-v1"),
                            max_len=int(src_hp.get("smiles_chunk_max_len", src_hp.get("chunk_max_len", 512))),
                            num_chunks=int(src_hp.get("smiles_num_chunks", src_hp.get("num_chunks", 8))),
                            dtype=chunk_dtype,
                        )
                    else:
                        compute_and_save_chem_embeddings(
                            [smiles], tmp_chem, src_hp.get("chem_model", "seyonec/ChemBERTa-zinc-base-v1")
                        )
                    with open(tmp_chem, "rb") as f:
                        chem_dict.update(pickle.load(f))

                if seq not in esm_dict:
                    tmp_esm = os.path.join(tmp_dir, "esm.pkl")
                    if use_chunked:
                        compute_and_save_chunked_embeddings(
                            [seq],
                            tmp_esm,
                            src_hp.get("esm_model", "esm2_t12_35M_UR50D"),
                            max_len=int(src_hp.get("protein_chunk_max_len", src_hp.get("chunk_max_len", 512))),
                            num_chunks=int(src_hp.get("protein_num_chunks", src_hp.get("num_chunks", 8))),
                            dtype=chunk_dtype,
                        )
                    else:
                        compute_and_save_embeddings([seq], tmp_esm, src_hp.get("esm_model", "esm2_t12_35M_UR50D"))
                    with open(tmp_esm, "rb") as f:
                        esm_dict.update(pickle.load(f))

                result = score_dtpi_inference(model_path, chem_dict, esm_dict, df)[0]
                return _single_result_payload(task_type, result)

            if task_type == "rpi":
                field_error = _require_payload_fields(payload, ["rna_sequence", "protein_sequence"])
                if field_error:
                    return JSONResponse(field_error, status_code=400)
                length_error = _require_max_lengths(payload, ["rna_sequence", "protein_sequence"])
                if length_error:
                    return JSONResponse(length_error, status_code=400)

                rna_seq = str(payload["rna_sequence"]).strip().upper().replace("T", "U")
                prot_seq = str(payload["protein_sequence"]).strip().upper()
                df = pd.DataFrame([{"rna_sequence": rna_seq, "protein_sequence": prot_seq}])

                rna_embed_path = os.path.join(src_dir, f"rna_embedding_{source_run_id}.pkl")
                esm_embed_path = os.path.join(src_dir, f"embedding_{source_run_id}.pkl")
                if not os.path.exists(rna_embed_path) or not os.path.exists(esm_embed_path):
                    return JSONResponse({"error": "Required source embeddings not found."}, status_code=404)
                with open(rna_embed_path, "rb") as f:
                    rna_dict = pickle.load(f)
                with open(esm_embed_path, "rb") as f:
                    esm_dict = pickle.load(f)

                if rna_seq not in rna_dict:
                    tmp_rna = os.path.join(tmp_dir, "rna.pkl")
                    if use_chunked:
                        compute_and_save_chunked_rna_embeddings(
                            [rna_seq],
                            tmp_rna,
                            src_hp.get("rna_model", "multimolecule/rnafm"),
                            max_len=int(src_hp.get("rna_chunk_max_len", src_hp.get("chunk_max_len", 512))),
                            num_chunks=int(src_hp.get("rna_num_chunks", src_hp.get("num_chunks", 8))),
                            dtype=chunk_dtype,
                        )
                    else:
                        compute_and_save_rna_embeddings([rna_seq], tmp_rna, src_hp.get("rna_model", "multimolecule/rnafm"))
                    with open(tmp_rna, "rb") as f:
                        rna_dict.update(pickle.load(f))

                if prot_seq not in esm_dict:
                    tmp_esm = os.path.join(tmp_dir, "esm.pkl")
                    if use_chunked:
                        compute_and_save_chunked_embeddings(
                            [prot_seq],
                            tmp_esm,
                            src_hp.get("esm_model", "esm2_t12_35M_UR50D"),
                            max_len=int(src_hp.get("protein_chunk_max_len", src_hp.get("chunk_max_len", 512))),
                            num_chunks=int(src_hp.get("protein_num_chunks", src_hp.get("num_chunks", 8))),
                            dtype=chunk_dtype,
                        )
                    else:
                        compute_and_save_embeddings([prot_seq], tmp_esm, src_hp.get("esm_model", "esm2_t12_35M_UR50D"))
                    with open(tmp_esm, "rb") as f:
                        esm_dict.update(pickle.load(f))

                result = score_rpi_inference(model_path, rna_dict, esm_dict, df)[0]
                return _single_result_payload(task_type, result)

            if task_type == "pdi":
                field_error = _require_payload_fields(payload, ["dna_sequence", "protein_sequence"])
                if field_error:
                    return JSONResponse(field_error, status_code=400)
                length_error = _require_max_lengths(payload, ["dna_sequence", "protein_sequence"])
                if length_error:
                    return JSONResponse(length_error, status_code=400)

                dna_seq = str(payload["dna_sequence"]).strip().upper()
                prot_seq = str(payload["protein_sequence"]).strip().upper()
                df = pd.DataFrame([{"dna_sequence": dna_seq, "protein_sequence": prot_seq}])

                dna_embed_path = os.path.join(src_dir, f"dna_embedding_{source_run_id}.pkl")
                esm_embed_path = os.path.join(src_dir, f"embedding_{source_run_id}.pkl")
                if not os.path.exists(dna_embed_path) or not os.path.exists(esm_embed_path):
                    return JSONResponse({"error": "Required source embeddings not found."}, status_code=404)
                with open(dna_embed_path, "rb") as f:
                    dna_dict = pickle.load(f)
                with open(esm_embed_path, "rb") as f:
                    esm_dict = pickle.load(f)

                if dna_seq not in dna_dict:
                    tmp_dna = os.path.join(tmp_dir, "dna.pkl")
                    if use_chunked:
                        compute_and_save_chunked_dna_embeddings(
                            [dna_seq],
                            tmp_dna,
                            src_hp.get("dna_model", "armheb/DNA_bert_6"),
                            max_len=int(src_hp.get("dna_chunk_max_len", src_hp.get("chunk_max_len", 512))),
                            num_chunks=int(src_hp.get("dna_num_chunks", src_hp.get("num_chunks", 8))),
                            dtype=chunk_dtype,
                        )
                    else:
                        compute_and_save_dna_embeddings([dna_seq], tmp_dna, src_hp.get("dna_model", "armheb/DNA_bert_6"))
                    with open(tmp_dna, "rb") as f:
                        dna_dict.update(pickle.load(f))

                if prot_seq not in esm_dict:
                    tmp_esm = os.path.join(tmp_dir, "esm.pkl")
                    if use_chunked:
                        compute_and_save_chunked_embeddings(
                            [prot_seq],
                            tmp_esm,
                            src_hp.get("esm_model", "esm2_t12_35M_UR50D"),
                            max_len=int(src_hp.get("protein_chunk_max_len", src_hp.get("chunk_max_len", 512))),
                            num_chunks=int(src_hp.get("protein_num_chunks", src_hp.get("num_chunks", 8))),
                            dtype=chunk_dtype,
                        )
                    else:
                        compute_and_save_embeddings([prot_seq], tmp_esm, src_hp.get("esm_model", "esm2_t12_35M_UR50D"))
                    with open(tmp_esm, "rb") as f:
                        esm_dict.update(pickle.load(f))

                result = score_pdi_inference(model_path, dna_dict, esm_dict, df)[0]
                return _single_result_payload(task_type, result)

            field_error = _require_payload_fields(payload, ["proteinA", "proteinB"])
            if field_error:
                return JSONResponse(field_error, status_code=400)
            length_error = _require_max_lengths(payload, ["proteinA", "proteinB"])
            if length_error:
                return JSONResponse(length_error, status_code=400)

            seq_a = str(payload["proteinA"]).strip().upper()
            seq_b = str(payload["proteinB"]).strip().upper()
            df = pd.DataFrame([{"proteinA": seq_a, "proteinB": seq_b}])

            embed_path = os.path.join(src_dir, f"embedding_{source_run_id}.pkl")
            if not os.path.exists(embed_path):
                return JSONResponse({"error": "Required source embeddings not found."}, status_code=404)
            with open(embed_path, "rb") as f:
                embedding_dict = pickle.load(f)

            new_seqs = sorted(({seq_a, seq_b} - set(embedding_dict.keys())) - {"NAN", ""})
            if new_seqs:
                tmp_esm = os.path.join(tmp_dir, "esm.pkl")
                if use_chunked:
                    compute_and_save_chunked_embeddings(
                        new_seqs,
                        tmp_esm,
                        src_hp.get("esm_model", "esm2_t12_35M_UR50D"),
                        max_len=int(src_hp.get("chunk_max_len", 512)),
                        num_chunks=int(src_hp.get("num_chunks", 8)),
                        dtype=chunk_dtype,
                    )
                else:
                    compute_and_save_embeddings(new_seqs, tmp_esm, src_hp.get("esm_model", "esm2_t12_35M_UR50D"))
                with open(tmp_esm, "rb") as f:
                    embedding_dict.update(pickle.load(f))

            result = score_ppi_inference(model_path, embedding_dict, df)[0]
            return _single_result_payload(task_type, result)

    except Exception as exc:
        return JSONResponse({"error": f"Single-pair inference failed: {exc}"}, status_code=500)


# ---------------------------------------------------------------------------
# POST /run_inference/{source_run_id}  — queue an inference job
# ---------------------------------------------------------------------------

@app.post("/run_inference/{source_run_id}")
async def create_inference_job(
    source_run_id: str,
    files: List[UploadFile] = File(...),
    infer_label: str = Form(""),
    is_single: bool = Form(False),
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
    try:
        paths = _save_uploaded_files(files, run_dir, run_id)
    except UploadTooLargeError as e:
        shutil.rmtree(run_dir, ignore_errors=True)
        return JSONResponse({"error": str(e)}, status_code=413)
    except ValueError as e:
        shutil.rmtree(run_dir, ignore_errors=True)
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        return JSONResponse({"error": "Could not process uploaded files."}, status_code=500)

    infer_hp = {
        "task_type":   src_task_type,
        "infer_label": infer_label.strip(),
        "is_single":   is_single,
    }
    db  = SessionLocal()
    try:
        job = Job(
            run_id         = run_id,
            status         = "queued",
            job_type       = "inference",
            input_sequence = json.dumps(paths),
            source_run_id  = source_run_id,
            hyperparams    = json.dumps(infer_hp),
        )
        db.add(job)
        db.commit()
    finally:
        db.close()

    try:
        if src_task_type == "dtpi":
            task = run_dtpi_inference_task.delay(run_id, source_run_id, paths)
        elif src_task_type == "rpi":
            task = run_rpi_inference_task.delay(run_id, source_run_id, paths)
        elif src_task_type == "pdi":
            task = run_pdi_inference_task.delay(run_id, source_run_id, paths)
        else:
            task = run_ppi_inference.delay(run_id, source_run_id, paths)
    except Exception as exc:
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.run_id == run_id).first()
            if job:
                job.status = "failed"
                job.result = f"Queue submission failed: {exc}"
                db.commit()
        finally:
            db.close()
        return _queue_unavailable_response(run_id)

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        job.celery_task_id = task.id
        db.commit()
    finally:
        db.close()

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
    Compute SHAP values for a completed inference run using KernelExplainer.
    Supports pooled PPI, DTPI, RPI, and PDI checkpoints. Inference-time
    embeddings are merged so runs with previously unseen inputs can be
    explained.
    """
    import pickle
    import numpy as np
    import pandas as pd
    import torch

    if not _valid_run_id(infer_run_id):
        return JSONResponse({"error": "invalid run_id"}, status_code=400)

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
        infer_input_files = json.loads(infer_job.input_sequence) if infer_job.input_sequence else []
    finally:
        db.close()

    task_type = str(src_hp.get("task_type", "ppi")).lower()
    layer_configs = src_hp.get("layer_configs", [
        {"type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3},
        {"type": "linear", "hidden_dim": 64,  "activation": "relu", "dropout": 0.2},
    ])

    src_dir    = os.path.join(MODELS_DIR, source_run_id)
    infer_dir  = os.path.join(MODELS_DIR, infer_run_id)
    model_path = os.path.join(src_dir, f"model_{source_run_id}.pt")

    if not os.path.exists(model_path):
        return JSONResponse({"error": "model checkpoint not found"}, status_code=404)

    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    representation_mode = str(ckpt.get("embedding_representation", "pooled")).lower()

    pair_mode = ckpt.get("pair_mode", "concat")
    if pair_mode not in ("concat", "product", "diff", "all"):
        pair_mode = "concat"

    task_cfg = {
        "ppi": {
            "left_path": f"embedding_{source_run_id}.pkl",
            "right_path": f"embedding_{source_run_id}.pkl",
            "left_new": f"new_embeddings_{infer_run_id}.pkl",
            "right_new": f"new_embeddings_{infer_run_id}.pkl",
            "left_col": "proteinA",
            "right_col": "proteinB",
            "left_norm": lambda v: str(v).strip().upper(),
            "right_norm": lambda v: str(v).strip().upper(),
            "left_dim": int(ckpt.get("esm_dim", src_hp.get("esm_dim", 480))),
            "right_dim": int(ckpt.get("esm_dim", src_hp.get("esm_dim", 480))),
            "left_label": "Protein A",
            "right_label": "Protein B",
        },
        "dtpi": {
            "left_path": f"chem_embedding_{source_run_id}.pkl",
            "right_path": f"embedding_{source_run_id}.pkl",
            "left_new": f"new_chem_{infer_run_id}.pkl",
            "right_new": f"new_esm_{infer_run_id}.pkl",
            "left_col": "smiles",
            "right_col": "sequence",
            "left_norm": lambda v: str(v).strip(),
            "right_norm": lambda v: str(v).strip().upper(),
            "left_dim": int(ckpt.get("chem_dim", src_hp.get("chem_dim", 768))),
            "right_dim": int(ckpt.get("esm_dim", src_hp.get("esm_dim", 480))),
            "left_label": "Compound",
            "right_label": "Protein",
        },
        "rpi": {
            "left_path": f"rna_embedding_{source_run_id}.pkl",
            "right_path": f"embedding_{source_run_id}.pkl",
            "left_new": f"new_rna_{infer_run_id}.pkl",
            "right_new": f"new_esm_{infer_run_id}.pkl",
            "left_col": "rna_sequence",
            "right_col": "protein_sequence",
            "left_norm": lambda v: str(v).strip().upper().replace("T", "U"),
            "right_norm": lambda v: str(v).strip().upper(),
            "left_dim": int(ckpt.get("rna_dim", src_hp.get("rna_dim", 640))),
            "right_dim": int(ckpt.get("esm_dim", src_hp.get("esm_dim", 480))),
            "left_label": "RNA",
            "right_label": "Protein",
        },
        "pdi": {
            "left_path": f"dna_embedding_{source_run_id}.pkl",
            "right_path": f"embedding_{source_run_id}.pkl",
            "left_new": f"new_dna_{infer_run_id}.pkl",
            "right_new": f"new_esm_{infer_run_id}.pkl",
            "left_col": "dna_sequence",
            "right_col": "protein_sequence",
            "left_norm": lambda v: str(v).strip().upper(),
            "right_norm": lambda v: str(v).strip().upper(),
            "left_dim": int(ckpt.get("dna_dim", src_hp.get("dna_dim", 768))),
            "right_dim": int(ckpt.get("esm_dim", src_hp.get("esm_dim", 480))),
            "left_label": "DNA",
            "right_label": "Protein",
        },
    }.get(task_type)

    if task_cfg is None:
        return JSONResponse({"error": f"unsupported task_type={task_type!r}"}, status_code=400)

    if representation_mode == "chunked":
        from model_build.sequence_models import FlexiblePPISequenceModel, FlexiblePairSequenceModel

        if task_type == "ppi":
            model = FlexiblePPISequenceModel(ckpt["input_dim"], ckpt.get("layer_configs", layer_configs))
        else:
            model = FlexiblePairSequenceModel(
                task_cfg["left_dim"],
                task_cfg["right_dim"],
                int(ckpt.get("chunk_model_dim", ckpt["input_dim"])),
                ckpt.get("layer_configs", layer_configs),
            )
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        from model_build.ppi_classifier import FlexiblePPIModel

        model = FlexiblePPIModel(ckpt["input_dim"], ckpt.get("layer_configs", layer_configs))
        model.load_state_dict(ckpt["model_state"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    def _load_dict(path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            return pickle.load(fh)

    left_dict = _load_dict(os.path.join(src_dir, task_cfg["left_path"]))
    right_dict = left_dict if task_cfg["right_path"] == task_cfg["left_path"] else _load_dict(os.path.join(src_dir, task_cfg["right_path"]))
    if left_dict is None:
        return JSONResponse({"error": f"{task_cfg['left_label']} embedding file not found"}, status_code=404)
    if right_dict is None:
        return JSONResponse({"error": f"{task_cfg['right_label']} embedding file not found"}, status_code=404)

    left_new = _load_dict(os.path.join(infer_dir, task_cfg["left_new"]))
    if left_new:
        left_dict.update(left_new)
    if task_cfg["right_new"] == task_cfg["left_new"]:
        right_dict = left_dict
    else:
        right_new = _load_dict(os.path.join(infer_dir, task_cfg["right_new"]))
        if right_new:
            right_dict.update(right_new)

    input_paths = [p for p in infer_input_files if os.path.exists(p)]
    if input_paths:
        df = pd.concat([pd.read_csv(p) for p in input_paths], ignore_index=True)
    else:
        results_csv = os.path.join(infer_dir, f"results_{infer_run_id}.csv")
        if not os.path.exists(results_csv):
            return JSONResponse({"error": "results CSV not found"}, status_code=404)
        df = pd.read_csv(results_csv)
        if "probability" in df.columns:
            df = df.dropna(subset=["probability"])

    def _as_tensor(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().float().reshape(-1)
        return torch.tensor(np.asarray(value, dtype=np.float32)).float().reshape(-1)

    rows = []
    left_shape = None
    right_shape = None
    left_values = (
        df[task_cfg["left_col"]].tolist()
        if task_cfg["left_col"] in df.columns else [""] * len(df)
    )
    right_values = (
        df[task_cfg["right_col"]].tolist()
        if task_cfg["right_col"] in df.columns else [""] * len(df)
    )
    for left_value, right_value in zip(left_values, right_values):
        left_key = task_cfg["left_norm"](left_value)
        right_key = task_cfg["right_norm"](right_value)
        eA = left_dict.get(left_key)
        eB = right_dict.get(right_key)
        if eA is not None and eB is not None:
            if representation_mode == "chunked":
                left_arr = eA.detach().cpu().float().numpy() if hasattr(eA, "detach") else np.asarray(eA, dtype=np.float32)
                right_arr = eB.detach().cpu().float().numpy() if hasattr(eB, "detach") else np.asarray(eB, dtype=np.float32)
                if left_arr.ndim != 2 or right_arr.ndim != 2:
                    continue
                if left_shape is None:
                    left_shape = left_arr.shape
                    right_shape = right_arr.shape
                if left_arr.shape != left_shape or right_arr.shape != right_shape:
                    continue
                vec = np.concatenate([left_arr.reshape(-1), right_arr.reshape(-1)])
            else:
                eA_f = _as_tensor(eA)
                eB_f = _as_tensor(eB)
                if pair_mode == "all":
                    vec = torch.cat([eA_f, eB_f, eA_f * eB_f, (eA_f - eB_f).abs()], dim=-1)
                elif pair_mode == "product":
                    vec = eA_f * eB_f
                elif pair_mode == "diff":
                    vec = (eA_f - eB_f).abs()
                else:
                    vec = torch.cat([eA_f, eB_f], dim=-1)
                vec = vec.numpy()
            rows.append(vec)
        if len(rows) >= n_explain + n_background:
            break

    if len(rows) < 4:
        return JSONResponse(
            {"error": f"too few embeddable pairs for SHAP ({len(rows)} found, need at least 4)"},
            status_code=400,
        )

    X = np.stack(rows)

    try:
        import shap

        if representation_mode == "chunked":
            if left_shape is None or right_shape is None:
                return JSONResponse({"error": "chunked SHAP inputs could not be shaped"}, status_code=400)
            left_size = int(np.prod(left_shape))
            right_size = int(np.prod(right_shape))

            def _predict(x_np):
                with torch.inference_mode():
                    arr = np.asarray(x_np, dtype=np.float32)
                    left_np = arr[:, :left_size].reshape((-1, *left_shape))
                    right_np = arr[:, left_size:left_size + right_size].reshape((-1, *right_shape))
                    left = torch.tensor(left_np, dtype=torch.float).to(device)
                    right = torch.tensor(right_np, dtype=torch.float).to(device)
                    left_mask = (left.abs().sum(dim=-1) > 0).to(device)
                    right_mask = (right.abs().sum(dim=-1) > 0).to(device)
                    if task_type == "ppi":
                        merged = torch.cat([left, right], dim=1)
                        merged_mask = torch.cat([left_mask, right_mask], dim=1)
                        logits = model(merged, merged_mask)
                    else:
                        logits = model(left, right, left_mask, right_mask)
                    return torch.sigmoid(logits).cpu().numpy()

        else:
            def _predict(x_np):
                with torch.inference_mode():
                    t = torch.tensor(x_np, dtype=torch.float).to(device)
                    return torch.sigmoid(model(t)).cpu().numpy()

        n_bg  = min(n_background, len(X) // 2)
        n_exp = min(n_explain, len(X) - n_bg)
        background = X[:n_bg]
        explain_X  = X[n_bg: n_bg + n_exp]

        explainer   = shap.KernelExplainer(_predict, background)
        shap_values = explainer.shap_values(explain_X, nsamples=128, silent=True)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap_arr = np.asarray(shap_values)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, -1]
        mean_abs = np.abs(shap_arr).mean(axis=0)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.mean(axis=-1)

    except ImportError:
        # shap not installed: fall back to gradient-free sensitivity.
        baseline = X.mean(axis=0, keepdims=True)
        mean_abs = np.zeros(X.shape[1])
        p_orig = None
        for d in range(X.shape[1]):
            perturbed      = baseline.copy()
            perturbed[0, d] += float(X[:, d].std() or 1e-3)
            with torch.inference_mode():
                if p_orig is None:
                    p_orig = torch.sigmoid(model(torch.tensor(baseline, dtype=torch.float).to(device))).item()
                p_pert = torch.sigmoid(model(torch.tensor(perturbed, dtype=torch.float).to(device))).item()
            mean_abs[d] = abs(p_pert - p_orig)

    if representation_mode == "chunked" and left_shape is not None and right_shape is not None:
        left_dim = int(np.prod(left_shape))
        right_dim = int(np.prod(right_shape))
    else:
        left_dim = task_cfg["left_dim"]
        right_dim = task_cfg["right_dim"]
    if pair_mode in ("concat", "all"):
        eA_shap = mean_abs[:left_dim]
        eB_shap = mean_abs[left_dim:left_dim + right_dim]
        split_dim = left_dim
    else:
        eA_shap = mean_abs
        eB_shap = mean_abs
        split_dim = len(mean_abs)

    def _top(arr, k=15):
        idx = np.argsort(arr)[::-1][:k]
        return [{"dim": int(i), "value": float(round(arr[i], 6))} for i in idx]

    return {
        "eA_mean":     float(round(eA_shap.mean(), 6)),
        "eB_mean":     float(round(eB_shap.mean(), 6)),
        "eA_top":      _top(eA_shap),
        "eB_top":      _top(eB_shap),
        "global_top":  _top(mean_abs),
        "all_dims":    mean_abs.tolist(),
        "esm_dim":     split_dim,
        "left_dim":    left_dim,
        "right_dim":   right_dim,
        "task_type":   task_type,
        "pair_mode":   pair_mode,
        "representation_mode": representation_mode,
        "left_label":  task_cfg["left_label"],
        "right_label": task_cfg["right_label"],
        "n_pairs":     len(rows),
    }



# ---------------------------------------------------------------------------
# GET /download_bundle/{run_id}  — zipped artifact bundle
# ---------------------------------------------------------------------------

@app.get("/download_bundle/{run_id}")
def download_bundle(run_id: str):
    if not _valid_run_id(run_id):
        return JSONResponse({"error": "invalid run_id"}, status_code=400)

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
    finally:
        db.close()

    if not job:
        return JSONResponse({"error": "run_id not found"}, status_code=404)
    if job.status != "completed":
        return JSONResponse({"error": "job not completed"}, status_code=400)

    try:
        zip_path, n_files = _build_artifact_bundle(run_id, job)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"bundle creation failed: {e}"}, status_code=500)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"artifacts_{run_id}.zip",
        headers={"X-Bundle-Files": str(n_files)},
    )


# ---------------------------------------------------------------------------
# GET /job_detail/{run_id}  — full job record including hyperparams
# ---------------------------------------------------------------------------

@app.get("/job_detail/{run_id}")
def job_detail(run_id: str):
    if not _valid_run_id(run_id):
        return JSONResponse({"error": "invalid run_id"}, status_code=400)
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

    def _as_numpy(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().float().numpy()
        return np.asarray(value, dtype=np.float32)

    def _protein_umap_vec(value):
        arr = _as_numpy(value)
        if arr.ndim == 1:
            return arr.astype(np.float32)
        if arr.ndim == 2:
            real_mask = np.abs(arr).sum(axis=1) > 0
            if real_mask.any():
                return arr[real_mask].mean(axis=0).astype(np.float32)
            return np.zeros(arr.shape[-1], dtype=np.float32)
        return arr.reshape(-1).astype(np.float32)

    def _load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def _label_value(value):
        try:
            return int(value) if str(value) not in ("nan", "", "None") else -1
        except Exception:
            return -1

    rows, labels, orig_idx = [], [], []
    label_values = df["label"].tolist() if "label" in df.columns else [""] * len(df)

    if task_type == "ppi":
        emb = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        protein_a = (
            df["proteinA"].astype(str).str.strip().str.upper().tolist()
            if "proteinA" in df.columns else [""] * len(df)
        )
        protein_b = (
            df["proteinB"].astype(str).str.strip().str.upper().tolist()
            if "proteinB" in df.columns else [""] * len(df)
        )
        for i, (a, b, label) in enumerate(zip(protein_a, protein_b, label_values)):
            if a in emb and b in emb:
                rows.append(np.concatenate([_protein_umap_vec(emb[a]), _protein_umap_vec(emb[b])]))
                labels.append(_label_value(label))
                orig_idx.append(i)

    elif task_type == "dtpi":
        chem = _load(os.path.join(run_dir, f"chem_embedding_{run_id}.pkl"))
        esm  = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        smiles_values = (
            df["smiles"].astype(str).str.strip().tolist()
            if "smiles" in df.columns else [""] * len(df)
        )
        protein_values = (
            df["sequence"].astype(str).str.strip().str.upper().tolist()
            if "sequence" in df.columns else [""] * len(df)
        )
        for i, (s, p, label) in enumerate(zip(smiles_values, protein_values, label_values)):
            if s in chem and p in esm:
                rows.append(np.concatenate([_protein_umap_vec(chem[s]), _protein_umap_vec(esm[p])]))
                labels.append(_label_value(label))
                orig_idx.append(i)

    elif task_type == "rpi":
        rna = _load(os.path.join(run_dir, f"rna_embedding_{run_id}.pkl"))
        esm = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        rna_values = (
            df["rna_sequence"].astype(str).str.strip().str.upper()
            .str.replace("T", "U", regex=False).tolist()
            if "rna_sequence" in df.columns else [""] * len(df)
        )
        protein_values = (
            df["protein_sequence"].astype(str).str.strip().str.upper().tolist()
            if "protein_sequence" in df.columns else [""] * len(df)
        )
        for i, (r, p, label) in enumerate(zip(rna_values, protein_values, label_values)):
            if r in rna and p in esm:
                rows.append(np.concatenate([_protein_umap_vec(rna[r]), _protein_umap_vec(esm[p])]))
                labels.append(_label_value(label))
                orig_idx.append(i)

    elif task_type == "pdi":
        dna = _load(os.path.join(run_dir, f"dna_embedding_{run_id}.pkl"))
        esm = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        dna_values = (
            df["dna_sequence"].astype(str).str.strip().str.upper().tolist()
            if "dna_sequence" in df.columns else [""] * len(df)
        )
        protein_values = (
            df["protein_sequence"].astype(str).str.strip().str.upper().tolist()
            if "protein_sequence" in df.columns else [""] * len(df)
        )
        for i, (d, p, label) in enumerate(zip(dna_values, protein_values, label_values)):
            if d in dna and p in esm:
                rows.append(np.concatenate([_protein_umap_vec(dna[d]), _protein_umap_vec(esm[p])]))
                labels.append(_label_value(label))
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

    def _label_value(value):
        try:
            return int(value) if str(value) not in ("nan", "", "None") else -1
        except Exception:
            return -1

    rows, labels, orig_idx = [], [], []
    label_values = df["label"].tolist() if "label" in df.columns else [""] * len(df)

    if task_type == "ppi":
        emb = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        protein_a = (
            df["proteinA"].astype(str).str.strip().str.upper().tolist()
            if "proteinA" in df.columns else [""] * len(df)
        )
        protein_b = (
            df["proteinB"].astype(str).str.strip().str.upper().tolist()
            if "proteinB" in df.columns else [""] * len(df)
        )
        for i, (a, b, label) in enumerate(zip(protein_a, protein_b, label_values)):
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
                labels.append(_label_value(label))
                orig_idx.append(i)

    elif task_type == "dtpi":
        chem = _load(os.path.join(run_dir, f"chem_embedding_{run_id}.pkl"))
        esm  = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        smiles_values = (
            df["smiles"].astype(str).str.strip().tolist()
            if "smiles" in df.columns else [""] * len(df)
        )
        protein_values = (
            df["sequence"].astype(str).str.strip().str.upper().tolist()
            if "sequence" in df.columns else [""] * len(df)
        )
        for i, (s, p, label) in enumerate(zip(smiles_values, protein_values, label_values)):
            if s in chem and p in esm:
                rows.append(np.concatenate([chem[s], esm[p]]))
                labels.append(_label_value(label))
                orig_idx.append(i)

    elif task_type == "rpi":
        rna = _load(os.path.join(run_dir, f"rna_embedding_{run_id}.pkl"))
        esm = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        rna_values = (
            df["rna_sequence"].astype(str).str.strip().str.upper()
            .str.replace("T", "U", regex=False).tolist()
            if "rna_sequence" in df.columns else [""] * len(df)
        )
        protein_values = (
            df["protein_sequence"].astype(str).str.strip().str.upper().tolist()
            if "protein_sequence" in df.columns else [""] * len(df)
        )
        for i, (r, p, label) in enumerate(zip(rna_values, protein_values, label_values)):
            if r in rna and p in esm:
                rows.append(np.concatenate([rna[r], esm[p]]))
                labels.append(_label_value(label))
                orig_idx.append(i)

    elif task_type == "pdi":
        dna = _load(os.path.join(run_dir, f"dna_embedding_{run_id}.pkl"))
        esm = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        dna_values = (
            df["dna_sequence"].astype(str).str.strip().str.upper().tolist()
            if "dna_sequence" in df.columns else [""] * len(df)
        )
        protein_values = (
            df["protein_sequence"].astype(str).str.strip().str.upper().tolist()
            if "protein_sequence" in df.columns else [""] * len(df)
        )
        for i, (d, p, label) in enumerate(zip(dna_values, protein_values, label_values)):
            if d in dna and p in esm:
                rows.append(np.concatenate([dna[d], esm[p]]))
                labels.append(_label_value(label))
                orig_idx.append(i)

    if not rows:
        return np.array([]), [], []
    return np.array(rows, dtype=np.float32), labels, orig_idx


def _build_chunked_ppi_model_inputs(run_id: str, run_dir: str, df) -> tuple:
    """Returns (chunk_tensor, mask_tensor, labels, orig_df_indices) for chunked PPI checkpoints."""
    import pickle
    import numpy as np

    def _load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def _as_numpy(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().float().numpy()
        return np.asarray(value, dtype=np.float32)

    def _chunk_mask(arr):
        return (np.abs(arr).sum(axis=-1) > 0).astype(bool)

    def _label_value(value):
        try:
            return int(value) if str(value) not in ("nan", "", "None") else -1
        except Exception:
            return -1

    emb = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
    rows, masks, labels, orig_idx = [], [], [], []
    label_values = df["label"].tolist() if "label" in df.columns else [""] * len(df)
    protein_a = (
        df["proteinA"].astype(str).str.strip().str.upper().tolist()
        if "proteinA" in df.columns else [""] * len(df)
    )
    protein_b = (
        df["proteinB"].astype(str).str.strip().str.upper().tolist()
        if "proteinB" in df.columns else [""] * len(df)
    )
    for i, (a, b, label) in enumerate(zip(protein_a, protein_b, label_values)):
        if a not in emb or b not in emb:
            continue

        eA = _as_numpy(emb[a])
        eB = _as_numpy(emb[b])
        if eA.ndim != 2 or eB.ndim != 2:
            continue

        rows.append(np.concatenate([eA, eB], axis=0).astype(np.float32))
        masks.append(np.concatenate([_chunk_mask(eA), _chunk_mask(eB)], axis=0))
        labels.append(_label_value(label))
        orig_idx.append(i)

    if not rows:
        return np.array([]), np.array([]), [], []
    return np.stack(rows).astype(np.float32), np.stack(masks).astype(bool), labels, orig_idx


def _build_chunked_pair_model_inputs(run_id: str, run_dir: str, task_type: str, df) -> tuple:
    """Returns left/right chunk tensors, masks, labels, and source row indices."""
    import pickle
    import numpy as np

    config = {
        "dtpi": {
            "left_path": f"chem_embedding_{run_id}.pkl",
            "right_path": f"embedding_{run_id}.pkl",
            "left_col": "smiles",
            "right_col": "sequence",
            "left_norm": lambda v: str(v).strip(),
            "right_norm": lambda v: str(v).strip().upper(),
        },
        "rpi": {
            "left_path": f"rna_embedding_{run_id}.pkl",
            "right_path": f"embedding_{run_id}.pkl",
            "left_col": "rna_sequence",
            "right_col": "protein_sequence",
            "left_norm": lambda v: str(v).strip().upper().replace("T", "U"),
            "right_norm": lambda v: str(v).strip().upper(),
        },
        "pdi": {
            "left_path": f"dna_embedding_{run_id}.pkl",
            "right_path": f"embedding_{run_id}.pkl",
            "left_col": "dna_sequence",
            "right_col": "protein_sequence",
            "left_norm": lambda v: str(v).strip().upper(),
            "right_norm": lambda v: str(v).strip().upper(),
        },
    }.get(task_type)
    if config is None:
        return np.array([]), np.array([]), np.array([]), np.array([]), [], []

    def _load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def _as_numpy(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().float().numpy()
        return np.asarray(value, dtype=np.float32)

    def _chunk_mask(arr):
        return (np.abs(arr).sum(axis=-1) > 0).astype(bool)

    def _label_value(value):
        try:
            return int(value) if str(value) not in ("nan", "", "None") else -1
        except Exception:
            return -1

    left_dict = _load(os.path.join(run_dir, config["left_path"]))
    right_dict = _load(os.path.join(run_dir, config["right_path"]))
    left_rows, right_rows, left_masks, right_masks, labels, orig_idx = [], [], [], [], [], []

    label_values = df["label"].tolist() if "label" in df.columns else [""] * len(df)
    left_values = (
        df[config["left_col"]].tolist()
        if config["left_col"] in df.columns else [""] * len(df)
    )
    right_values = (
        df[config["right_col"]].tolist()
        if config["right_col"] in df.columns else [""] * len(df)
    )
    for i, (left_value, right_value, label) in enumerate(zip(left_values, right_values, label_values)):
        left_key = config["left_norm"](left_value)
        right_key = config["right_norm"](right_value)
        if left_key not in left_dict or right_key not in right_dict:
            continue
        left = _as_numpy(left_dict[left_key])
        right = _as_numpy(right_dict[right_key])
        if left.ndim != 2 or right.ndim != 2:
            continue
        left_rows.append(left.astype(np.float32))
        right_rows.append(right.astype(np.float32))
        left_masks.append(_chunk_mask(left))
        right_masks.append(_chunk_mask(right))
        labels.append(_label_value(label))
        orig_idx.append(i)

    if not left_rows:
        return np.array([]), np.array([]), np.array([]), np.array([]), [], []
    return (
        np.stack(left_rows).astype(np.float32),
        np.stack(right_rows).astype(np.float32),
        np.stack(left_masks).astype(bool),
        np.stack(right_masks).astype(bool),
        labels,
        orig_idx,
    )


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
        "dtpi": [("smiles", "SMILES"), ("sequence", "Protein")],
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
        from model_build.sequence_models import FlexiblePPISequenceModel, FlexiblePairSequenceModel

        ckpt         = torch.load(model_path, map_location="cpu", weights_only=True)
        representation_mode = ckpt.get("embedding_representation", "pooled")
        input_dim    = ckpt["input_dim"]
        layer_configs = ckpt["layer_configs"]
        pair_mode    = ckpt.get("pair_mode", "concat")

        if representation_mode == "chunked":
            if task_type == "ppi":
                model = FlexiblePPISequenceModel(input_dim, layer_configs)
            else:
                dim_key = {"dtpi": "chem_dim", "rpi": "rna_dim", "pdi": "dna_dim"}.get(task_type)
                if dim_key is None:
                    return JSONResponse({"error": "unsupported task type"}, status_code=400)
                model = FlexiblePairSequenceModel(
                    int(ckpt.get(dim_key, hp.get(dim_key, 768 if task_type != "rpi" else 640))),
                    int(ckpt.get("esm_dim", hp.get("esm_dim", 480))),
                    int(ckpt.get("chunk_model_dim", input_dim)),
                    layer_configs,
                )
            model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            model = FlexiblePPIModel(input_dim, layer_configs)
            model.load_state_dict(ckpt["model_state"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        run_dir = os.path.join(MODELS_DIR, run_id)
        L = R = LM = RM = None
        if representation_mode == "chunked":
            if task_type == "ppi":
                X, M, true_labels, orig_indices = _build_chunked_ppi_model_inputs(run_id, run_dir, df)
            else:
                L, R, LM, RM, true_labels, orig_indices = _build_chunked_pair_model_inputs(run_id, run_dir, task_type, df)
                X = L
                M = None
        else:
            X, true_labels, orig_indices = _build_model_inputs(run_id, run_dir, task_type, df, pair_mode)
            M = None

        # keep only validation samples
        val_set    = _reconstruct_val_set(df)
        val_mask   = [i for i, oi in enumerate(orig_indices) if oi in val_set]
        X          = X[val_mask]
        if M is not None:
            M = M[val_mask]
        if L is not None:
            L = L[val_mask]
            R = R[val_mask]
            LM = LM[val_mask]
            RM = RM[val_mask]
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
            if M is not None:
                M = M[chosen]
            if L is not None:
                L = L[chosen]
                R = R[chosen]
                LM = LM[chosen]
                RM = RM[chosen]
            true_labels = [true_labels[i] for i in chosen]

        # hook into output layer to capture penultimate activations
        penultimate: list = []
        def _hook(module, inp, out):
            penultimate.append(inp[0].detach().cpu().numpy())
        handle = model.output.register_forward_hook(_hook)

        probs_all: list = []
        X_tensor = torch.tensor(X, dtype=torch.float32)
        M_tensor = torch.tensor(M, dtype=torch.bool) if M is not None else None
        L_tensor = torch.tensor(L, dtype=torch.float32) if L is not None else None
        R_tensor = torch.tensor(R, dtype=torch.float32) if R is not None else None
        LM_tensor = torch.tensor(LM, dtype=torch.bool) if LM is not None else None
        RM_tensor = torch.tensor(RM, dtype=torch.bool) if RM is not None else None
        with torch.inference_mode():
            data_len = len(L_tensor) if L_tensor is not None else len(X_tensor)
            for i in range(0, data_len, 256):
                if L_tensor is not None:
                    left = L_tensor[i : i + 256].to(device)
                    right = R_tensor[i : i + 256].to(device)
                    left_mask = LM_tensor[i : i + 256].to(device)
                    right_mask = RM_tensor[i : i + 256].to(device)
                    logits = model(left, right, left_mask, right_mask)
                else:
                    batch  = X_tensor[i : i + 256].to(device)
                    if M_tensor is not None:
                        mask = M_tensor[i : i + 256].to(device)
                        logits = model(batch, mask)
                    else:
                        logits = model(batch)
                probs_all.extend(torch.sigmoid(logits).cpu().numpy().tolist())

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
def list_jobs(
    status: str | None = Query(default=None, description="Comma-separated statuses"),
    job_type: str | None = Query(default=None, description="train|inference"),
    task_type: str | None = Query(default=None, description="ppi|dtpi|rpi|pdi"),
    run_id_contains: str | None = Query(default=None, description="Substring match for run_id"),
    limit: int = Query(default=200, ge=1, le=1000, description="Max rows to return"),
    offset: int = Query(default=0, ge=0, description="Rows to skip"),
):
    db   = SessionLocal()
    try:
        q = db.query(Job)

        statuses: list[str] = []
        if status:
            statuses = [s.strip().lower() for s in status.split(",") if s.strip()]
            if statuses:
                q = q.filter(Job.status.in_(statuses))

        if job_type:
            q = q.filter(Job.job_type == job_type.strip().lower())

        if run_id_contains:
            rid = run_id_contains.strip()
            if rid:
                q = q.filter(Job.run_id.ilike(f"%{rid}%"))

        jobs = q.order_by(Job.created_at.desc()).all()
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

            row = {
                "run_id":        j.run_id,
                "status":        j.status,
                "job_type":      j.job_type or "train",
                "task_type":     hp_raw.get("task_type", "ppi"),
                "created_at":    j.created_at.isoformat() if j.created_at else None,
                "val_acc":       _safe(metrics.get("val_acc")),
                "auroc":         _safe(metrics.get("auroc")),
                "f1":            _safe(metrics.get("f1")),
                "trainable_params": metrics.get("trainable_params"),
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
                "embedding_representation": hp_raw.get("embedding_representation", hp_raw.get("representation_mode")),
                "representation_mode": hp_raw.get("representation_mode", hp_raw.get("embedding_representation")),
                "chunk_model_dim": hp_raw.get("chunk_model_dim"),
                "layer_configs": hp_raw.get("layer_configs", []),
                "epochs":        hp_raw.get("epochs"),
                "train_split":   hp_raw.get("train_split"),
            }

            if task_type and row["task_type"] != task_type.strip().lower():
                continue

            out.append(row)

        return out[offset: offset + limit]
    finally:
        db.close()
