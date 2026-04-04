import json
import os
import pickle
import traceback

import pandas as pd
from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded

from database import SessionLocal
from models import Job
from model_build.esm_embed import compute_and_save_embeddings, load_all_sequences
from model_build.ppi_classifier import train_classifier
from model_build.ppi_infer import run_inference

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

celery = Celery(
    "ppi_tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/1",
)

# Hard wall-clock limits per task.
# soft_time_limit raises SoftTimeLimitExceeded (caught → status=failed).
# time_limit sends SIGKILL after an additional 60 s grace period.
celery.conf.task_soft_time_limit = 4 * 3600   # 4 hours → graceful
celery.conf.task_time_limit      = 4 * 3600 + 60  # then hard kill

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MODELS_DIR = "/app/saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)


def _run_dir(run_id: str) -> str:
    path = os.path.join(MODELS_DIR, run_id)
    os.makedirs(path, exist_ok=True)
    return path


def _set_status(db, job, status: str):
    job.status = status
    db.commit()


# ---------------------------------------------------------------------------
# Task 1 — Training
# ---------------------------------------------------------------------------

@celery.task(name="train_ppi_model")
def train_ppi_model(run_id: str, input_files: list, hyperparams_json: str = "{}"):
    db = SessionLocal()
    job = db.query(Job).filter(Job.run_id == run_id).first()
    _set_status(db, job, "running")

    try:
        hyperparams   = json.loads(hyperparams_json)
        run_dir       = _run_dir(run_id)
        embed_path    = os.path.join(run_dir, f"embedding_{run_id}.pkl")
        model_path    = os.path.join(run_dir, f"model_{run_id}.pt")
        metrics_path  = os.path.join(run_dir, f"metrics_{run_id}.json")

        print(f"[{run_id}] Loading sequences from {input_files}", flush=True)
        seqs = load_all_sequences(input_files)
        print(f"[{run_id}] Unique sequences: {len(seqs)}", flush=True)

        # Step 1 — ESM2 embeddings
        compute_and_save_embeddings(all_sequences=seqs, outfile=embed_path)
        print(f"[{run_id}] Embeddings saved → {embed_path}", flush=True)

        # Step 2+3+4 — Pair representation + MLP training + save
        with open(embed_path, "rb") as f:
            embedding_dict = pickle.load(f)

        df = pd.concat([pd.read_csv(p) for p in input_files], ignore_index=True)

        final_metrics = train_classifier(
            df            = df,
            embedding_dict = embedding_dict,
            hyperparams   = hyperparams,
            metrics_path  = metrics_path,
            model_path    = model_path,
        )

        # Update DB
        job.status     = "completed"
        job.model_path = model_path
        job.metrics    = json.dumps(final_metrics)
        db.commit()
        print(f"[{run_id}] Training complete", flush=True)

    except SoftTimeLimitExceeded:
        print(f"[{run_id}] Hit 4-hour time limit — marking failed.", flush=True)
        job.status = "failed"
        job.result = "Job exceeded the 4-hour time limit and was automatically stopped."
        db.commit()

    except Exception as e:
        print(f"[{run_id}] ERROR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        job.status = "failed"
        job.result = str(e)
        db.commit()
        raise

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 2 — Inference
# ---------------------------------------------------------------------------

@celery.task(name="run_ppi_inference")
def run_ppi_inference(run_id: str, source_run_id: str, input_files: list):
    db  = SessionLocal()
    job = db.query(Job).filter(Job.run_id == run_id).first()
    _set_status(db, job, "running")

    try:
        run_dir     = _run_dir(run_id)
        results_csv = os.path.join(run_dir, f"results_{run_id}.csv")

        # Load source training run artefacts
        src_dir    = os.path.join(MODELS_DIR, source_run_id)
        embed_path = os.path.join(src_dir, f"embedding_{source_run_id}.pkl")
        model_path = os.path.join(src_dir, f"model_{source_run_id}.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")

        with open(embed_path, "rb") as f:
            embedding_dict = pickle.load(f)

        df = pd.concat([pd.read_csv(p) for p in input_files], ignore_index=True)

        # Check for sequences not yet embedded
        all_seqs = set(df["proteinA"].str.strip().str.upper()) | set(df["proteinB"].str.strip().str.upper())
        new_seqs = sorted(all_seqs - set(embedding_dict.keys()))

        if new_seqs:
            print(f"[{run_id}] Generating embeddings for {len(new_seqs)} new sequences", flush=True)
            tmp_embed_path = os.path.join(run_dir, f"new_embeddings_{run_id}.pkl")
            compute_and_save_embeddings(all_sequences=new_seqs, outfile=tmp_embed_path)
            with open(tmp_embed_path, "rb") as f:
                new_embeddings = pickle.load(f)
            embedding_dict.update(new_embeddings)

        results = run_inference(model_path, embedding_dict, df)

        # Save results as CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_csv, index=False)
        print(f"[{run_id}] Inference complete → {results_csv}", flush=True)

        job.status = "completed"
        job.result = results_csv
        db.commit()

    except Exception as e:
        print(f"[{run_id}] ERROR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        job.status = "failed"
        job.result = str(e)
        db.commit()
        raise

    finally:
        db.close()
