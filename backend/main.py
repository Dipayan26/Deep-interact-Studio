import hashlib
import json
import math
import os
import re
import secrets
import shutil
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

MODELS_DIR = "/app/saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)


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

RATE_LIMIT_CREATE_JOB_PER_MIN    = _int_env("RATE_LIMIT_CREATE_JOB_PER_MIN", 10)
RATE_LIMIT_RUN_INFERENCE_PER_MIN = _int_env("RATE_LIMIT_RUN_INFERENCE_PER_MIN", 20)
RATE_LIMIT_WINDOW_SECONDS = 60
_UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_MODEL_PARAMS = 5_000_000


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


def _is_upload_endpoint(method: str, path: str) -> bool:
    if method != "POST":
        return False
    return path == "/create_job" or path.startswith("/run_inference/")


def _rate_limit_for(method: str, path: str) -> tuple[int, str] | None:
    if method != "POST":
        return None
    if path == "/create_job":
        return RATE_LIMIT_CREATE_JOB_PER_MIN, "/create_job"
    if path.startswith("/run_inference/"):
        return RATE_LIMIT_RUN_INFERENCE_PER_MIN, "/run_inference/{source_run_id}"
    return None


def _client_identifier(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip() if forwarded else ""
    if first:
        return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


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


def _run_cleanup(failed_ttl_days: int = 1, completed_ttl_days: int = 7):
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

    if task_type == "dtpi":
        task = train_dtpi_model.delay(run_id, paths, json.dumps(hp))
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

    return {"run_id": run_id, "cancel_token": cancel_token, "leakage_warnings": leakage_warnings}


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

    if src_task_type == "dtpi":
        run_dtpi_inference_task.delay(run_id, source_run_id, paths)
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

    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    if ckpt.get("embedding_representation", "pooled") == "chunked":
        return JSONResponse(
            {"error": "SHAP is not supported for chunked PPI checkpoints yet."},
            status_code=400,
        )
    model = FlexiblePPIModel(ckpt["input_dim"], ckpt.get("layer_configs", layer_configs))
    model.load_state_dict(ckpt["model_state"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    with open(embed_path, "rb") as f:
        emb_dict = pickle.load(f)

    # Also load any new embeddings generated during inference for unseen sequences
    new_embed_path = os.path.join(MODELS_DIR, infer_run_id, f"new_embeddings_{infer_run_id}.pkl")
    if os.path.exists(new_embed_path):
        with open(new_embed_path, "rb") as f:
            emb_dict.update(pickle.load(f))

    pair_mode = ckpt.get("pair_mode", "concat")
    if pair_mode not in ("concat", "product", "diff", "all"):
        pair_mode = "concat"

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
            eA_f = eA.float()
            eB_f = eB.float()
            if pair_mode == "all":
                vec = torch.cat([eA_f, eB_f, eA_f * eB_f, (eA_f - eB_f).abs()], dim=-1)
            elif pair_mode == "product":
                vec = eA_f * eB_f
            elif pair_mode == "diff":
                vec = (eA_f - eB_f).abs()
            else:
                vec = torch.cat([eA_f, eB_f], dim=-1)
            rows.append(vec.numpy())
        if len(rows) >= n_explain + n_background:
            break

    if len(rows) < 4:
        return JSONResponse({"error": "too few embeddable pairs for SHAP"}, status_code=400)

    X = np.stack(rows)

    # ── KernelExplainer (model-agnostic) ─────────────────────────────────
    try:
        import shap

        def _predict(x_np):
            with torch.no_grad():
                t = torch.tensor(x_np, dtype=torch.float).to(device)
                return torch.sigmoid(model(t)).cpu().numpy()

        n_bg  = min(n_background, len(X) // 2)
        n_exp = min(n_explain, len(X) - n_bg)
        background = X[:n_bg]
        explain_X  = X[n_bg: n_bg + n_exp]

        explainer   = shap.KernelExplainer(_predict, background)
        shap_values = explainer.shap_values(explain_X, nsamples=128, silent=True)
        mean_abs    = np.abs(shap_values).mean(axis=0)

    except ImportError:
        # shap not installed — fall back to gradient-free sensitivity
        baseline = X.mean(axis=0, keepdims=True)
        mean_abs = np.zeros(X.shape[1])
        for d in range(X.shape[1]):
            perturbed      = baseline.copy()
            perturbed[0, d] = baseline[0, d] + baseline[0, d].std() + 1e-6
            with torch.no_grad():
                p_orig = torch.sigmoid(model(torch.tensor(baseline, dtype=torch.float))).item()
                p_pert = torch.sigmoid(model(torch.tensor(perturbed, dtype=torch.float))).item()
            mean_abs[d] = abs(p_pert - p_orig)

    # ── aggregate into feature groups ─────────────────────────────────────
    # For concat/all modes, first esm_dim positions correspond to protein A.
    # For product/diff, the vector is a single esm_dim vector (no clean A/B split).
    if pair_mode in ("concat", "all"):
        eA_shap = mean_abs[:esm_dim]
        eB_shap = mean_abs[esm_dim:2 * esm_dim]
    else:
        eA_shap = mean_abs
        eB_shap = mean_abs

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
# GET /shap/{infer_run_id}  — SHAP feature-group importances for inference run
# ---------------------------------------------------------------------------

@app.get("/shap/{infer_run_id}")
def get_shap(infer_run_id: str, n_background: int = 50, n_explain: int = 100):
    """
    Compute SHAP values for a PPI inference run using KernelExplainer.
    Falls back to gradient-free sensitivity analysis if shap is not installed.
    Only supported for PPI runs (single embedding dict).
    """
    import pickle
    import numpy as np
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
    finally:
        db.close()

    task_type = src_hp.get("task_type", "ppi")
    if task_type != "ppi":
        return JSONResponse(
            {"error": f"SHAP is currently only supported for PPI runs (got task_type={task_type!r})"},
            status_code=400,
        )

    esm_dim       = int(src_hp.get("esm_dim", 480))
    layer_configs = src_hp.get("layer_configs", [
        {"type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3},
        {"type": "linear", "hidden_dim": 64,  "activation": "relu", "dropout": 0.2},
    ])

    src_dir    = os.path.join(MODELS_DIR, source_run_id)
    model_path = os.path.join(src_dir, f"model_{source_run_id}.pt")
    embed_path = os.path.join(src_dir, f"embedding_{source_run_id}.pkl")

    if not os.path.exists(model_path):
        return JSONResponse({"error": "model checkpoint not found"}, status_code=404)
    if not os.path.exists(embed_path):
        return JSONResponse({"error": "embedding file not found"}, status_code=404)

    from model_build.ppi_classifier import FlexiblePPIModel

    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    if ckpt.get("embedding_representation", "pooled") == "chunked":
        return JSONResponse(
            {"error": "SHAP is not supported for chunked PPI checkpoints yet."},
            status_code=400,
        )
    model = FlexiblePPIModel(ckpt["input_dim"], ckpt.get("layer_configs", layer_configs))
    model.load_state_dict(ckpt["model_state"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    with open(embed_path, "rb") as f:
        emb_dict = pickle.load(f)

    results_csv = os.path.join(MODELS_DIR, infer_run_id, f"results_{infer_run_id}.csv")
    if not os.path.exists(results_csv):
        return JSONResponse({"error": "results CSV not found"}, status_code=404)

    import pandas as pd
    df   = pd.read_csv(results_csv).dropna(subset=["probability"])
    rows = []
    for _, row in df.iterrows():
        eA = emb_dict.get(str(row.get("proteinA", "")).strip().upper())
        eB = emb_dict.get(str(row.get("proteinB", "")).strip().upper())
        if eA is not None and eB is not None:
            rows.append(torch.cat([eA, eB], dim=-1).float().numpy())
        if len(rows) >= n_explain + n_background:
            break

    if len(rows) < 4:
        return JSONResponse({"error": "too few embeddable pairs for SHAP"}, status_code=400)

    X = np.stack(rows)

    try:
        import shap

        def _predict(x_np):
            with torch.no_grad():
                t = torch.tensor(x_np, dtype=torch.float).to(device)
                return torch.sigmoid(model(t)).cpu().numpy()

        n_bg  = min(n_background, len(X) // 2)
        n_exp = min(n_explain, len(X) - n_bg)
        explainer   = shap.KernelExplainer(_predict, X[:n_bg])
        shap_values = explainer.shap_values(X[n_bg: n_bg + n_exp], nsamples=128, silent=True)
        mean_abs    = np.abs(shap_values).mean(axis=0)

    except ImportError:
        baseline = X.mean(axis=0, keepdims=True)
        mean_abs = np.zeros(X.shape[1])
        for d in range(X.shape[1]):
            perturbed = baseline.copy()
            perturbed[0, d] += (baseline[0, d].std() if baseline[0, d].std() > 0 else 1e-3)
            with torch.no_grad():
                p0 = torch.sigmoid(model(torch.tensor(baseline, dtype=torch.float).to(device))).item()
                p1 = torch.sigmoid(model(torch.tensor(perturbed, dtype=torch.float).to(device))).item()
            mean_abs[d] = abs(p1 - p0)

    eA_shap = mean_abs[:esm_dim]
    eB_shap = mean_abs[esm_dim:]

    def _top(arr, k=15):
        idx = np.argsort(arr)[::-1][:k]
        return [{"dim": int(i), "value": float(round(arr[i], 6))} for i in idx]

    return {
        "eA_mean":    float(round(eA_shap.mean(), 6)),
        "eB_mean":    float(round(eB_shap.mean(), 6)),
        "eA_top":     _top(eA_shap),
        "eB_top":     _top(eB_shap),
        "global_top": _top(mean_abs),
        "all_dims":   mean_abs.tolist(),
        "esm_dim":    esm_dim,
    }


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
                rows.append(np.concatenate([_protein_umap_vec(emb[a]), _protein_umap_vec(emb[b])]))
                labels.append(_label(row))
                orig_idx.append(i)

    elif task_type == "dtpi":
        chem = _load(os.path.join(run_dir, f"chem_embedding_{run_id}.pkl"))
        esm  = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        for i, (_, row) in enumerate(df.iterrows()):
            s = str(row.get("smiles", "")).strip()
            p = str(row.get("sequence", "")).strip().upper()
            if s in chem and p in esm:
                rows.append(np.concatenate([_protein_umap_vec(chem[s]), _protein_umap_vec(esm[p])]))
                labels.append(_label(row))
                orig_idx.append(i)

    elif task_type == "rpi":
        rna = _load(os.path.join(run_dir, f"rna_embedding_{run_id}.pkl"))
        esm = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        for i, (_, row) in enumerate(df.iterrows()):
            r = str(row.get("rna_sequence", "")).strip().upper().replace("T", "U")
            p = str(row.get("protein_sequence", "")).strip().upper()
            if r in rna and p in esm:
                rows.append(np.concatenate([_protein_umap_vec(rna[r]), _protein_umap_vec(esm[p])]))
                labels.append(_label(row))
                orig_idx.append(i)

    elif task_type == "pdi":
        dna = _load(os.path.join(run_dir, f"dna_embedding_{run_id}.pkl"))
        esm = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
        for i, (_, row) in enumerate(df.iterrows()):
            d = str(row.get("dna_sequence", "")).strip().upper()
            p = str(row.get("protein_sequence", "")).strip().upper()
            if d in dna and p in esm:
                rows.append(np.concatenate([_protein_umap_vec(dna[d]), _protein_umap_vec(esm[p])]))
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

    elif task_type == "dtpi":
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

    def _label(row):
        try:
            v = row.get("label", "")
            return int(v) if str(v) not in ("nan", "", "None") else -1
        except Exception:
            return -1

    emb = _load(os.path.join(run_dir, f"embedding_{run_id}.pkl"))
    rows, masks, labels, orig_idx = [], [], [], []
    for i, (_, row) in enumerate(df.iterrows()):
        a = str(row.get("proteinA", "")).strip().upper()
        b = str(row.get("proteinB", "")).strip().upper()
        if a not in emb or b not in emb:
            continue

        eA = _as_numpy(emb[a])
        eB = _as_numpy(emb[b])
        if eA.ndim != 2 or eB.ndim != 2:
            continue

        rows.append(np.concatenate([eA, eB], axis=0).astype(np.float32))
        masks.append(np.concatenate([_chunk_mask(eA), _chunk_mask(eB)], axis=0))
        labels.append(_label(row))
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

    def _label(row):
        try:
            v = row.get("label", "")
            return int(v) if str(v) not in ("nan", "", "None") else -1
        except Exception:
            return -1

    left_dict = _load(os.path.join(run_dir, config["left_path"]))
    right_dict = _load(os.path.join(run_dir, config["right_path"]))
    left_rows, right_rows, left_masks, right_masks, labels, orig_idx = [], [], [], [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        left_key = config["left_norm"](row.get(config["left_col"], ""))
        right_key = config["right_norm"](row.get(config["right_col"], ""))
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
        labels.append(_label(row))
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
        with torch.no_grad():
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
            }

            if task_type and row["task_type"] != task_type.strip().lower():
                continue

            out.append(row)

        return out[offset: offset + limit]
    finally:
        db.close()
