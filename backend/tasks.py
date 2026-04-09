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
celery.conf.task_soft_time_limit = 4 * 3600        # 4 hours → graceful
celery.conf.task_time_limit      = 4 * 3600 + 60   # then hard kill

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
    db  = SessionLocal()
    job = db.query(Job).filter(Job.run_id == run_id).first()
    _set_status(db, job, "running")

    try:
        hyperparams  = json.loads(hyperparams_json)
        run_dir      = _run_dir(run_id)
        embed_path   = os.path.join(run_dir, f"embedding_{run_id}.pkl")
        model_path   = os.path.join(run_dir, f"model_{run_id}.pt")
        metrics_path = os.path.join(run_dir, f"metrics_{run_id}.json")

        # Determine which ESM2 model to use
        esm_model = hyperparams.get("esm_model", "esm2_t12_35M_UR50D")

        print(f"[{run_id}] Loading sequences from {input_files}", flush=True)
        seqs = load_all_sequences(input_files)
        print(f"[{run_id}] Unique sequences: {len(seqs)}  ESM model: {esm_model}", flush=True)

        # Step 1 — ESM2 embeddings
        compute_and_save_embeddings(
            all_sequences=seqs,
            outfile=embed_path,
            model_name=esm_model,
        )
        print(f"[{run_id}] Embeddings saved → {embed_path}", flush=True)

        # Step 2+3+4 — Pair representation + training + save
        with open(embed_path, "rb") as f:
            embedding_dict = pickle.load(f)

        df = pd.concat([pd.read_csv(p) for p in input_files], ignore_index=True)

        final_metrics = train_classifier(
            df             = df,
            embedding_dict = embedding_dict,
            hyperparams    = hyperparams,
            metrics_path   = metrics_path,
            model_path     = model_path,
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

        # Fetch source job hyperparams to get the ESM model used during training
        src_db = SessionLocal()
        try:
            src_job = src_db.query(Job).filter(Job.run_id == source_run_id).first()
            src_hp  = json.loads(src_job.hyperparams) if src_job and src_job.hyperparams else {}
        finally:
            src_db.close()

        esm_model = src_hp.get("esm_model", "esm2_t12_35M_UR50D")

        # Load source training run artefacts
        src_dir    = os.path.join(MODELS_DIR, source_run_id)
        embed_path = os.path.join(src_dir, f"embedding_{source_run_id}.pkl")
        model_path = os.path.join(src_dir, f"model_{source_run_id}.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        if not os.path.exists(embed_path):
            raise FileNotFoundError(f"Embedding file not found: {embed_path}")

        with open(embed_path, "rb") as f:
            embedding_dict = pickle.load(f)

        df = pd.concat([pd.read_csv(p) for p in input_files], ignore_index=True)

        # Embed any sequences not already in the embedding dict
        all_seqs = (
            set(df["proteinA"].astype(str).str.strip().str.upper())
            | set(df["proteinB"].astype(str).str.strip().str.upper())
        )
        new_seqs = sorted(all_seqs - set(embedding_dict.keys()))

        if new_seqs:
            print(
                f"[{run_id}] Generating embeddings for {len(new_seqs)} new sequences "
                f"using {esm_model}",
                flush=True,
            )
            tmp_embed_path = os.path.join(run_dir, f"new_embeddings_{run_id}.pkl")
            compute_and_save_embeddings(
                all_sequences=new_seqs,
                outfile=tmp_embed_path,
                model_name=esm_model,
            )
            with open(tmp_embed_path, "rb") as f:
                new_embeddings = pickle.load(f)
            embedding_dict.update(new_embeddings)

        results = run_inference(model_path, embedding_dict, df)

        # Save results as CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_csv, index=False)
        print(f"[{run_id}] Inference complete → {results_csv}", flush=True)

        # ── Rich inference metrics for the dashboard ──────────────────────
        probs_list = [
            r["probability"] for r in results if r.get("probability") is not None
        ]
        has_labels = "label" in df.columns
        infer_metrics: dict = {"has_labels": has_labels, "probabilities": probs_list}

        if has_labels and probs_list:
            import math as _math
            import numpy as _np
            from sklearn.metrics import (
                roc_auc_score, average_precision_score,
                f1_score, accuracy_score, matthews_corrcoef,
            )

            valid_idx = [
                i for i, r in enumerate(results) if r.get("probability") is not None
            ]
            y_true = df["label"].astype(int).iloc[valid_idx].tolist()
            y_pred = [1 if p >= 0.5 else 0 for p in probs_list]
            infer_metrics["labels"] = y_true

            def _sf(v):
                try:
                    f = float(v)
                    return None if (_math.isnan(f) or _math.isinf(f)) else round(f, 4)
                except Exception:
                    return None

            try:
                infer_metrics["auroc"]    = _sf(roc_auc_score(y_true, probs_list))
                infer_metrics["auprc"]    = _sf(average_precision_score(y_true, probs_list))
                infer_metrics["f1"]       = _sf(f1_score(y_true, y_pred, zero_division=0))
                infer_metrics["accuracy"] = _sf(accuracy_score(y_true, y_pred))
                infer_metrics["mcc"]      = _sf(matthews_corrcoef(y_true, y_pred))
            except Exception as metric_err:
                print(f"[{run_id}] Metric computation skipped: {metric_err}", flush=True)

        infer_metrics_path = os.path.join(run_dir, f"infer_metrics_{run_id}.json")
        with open(infer_metrics_path, "w") as f:
            json.dump(infer_metrics, f)
        print(f"[{run_id}] Inference metrics saved → {infer_metrics_path}", flush=True)
        # ─────────────────────────────────────────────────────────────────

        job.status = "completed"
        job.result = results_csv
        db.commit()

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
