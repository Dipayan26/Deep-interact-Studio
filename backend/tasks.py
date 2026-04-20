import json
import os
import pickle
import shutil
import traceback

import pandas as pd
from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded

from database import SessionLocal
from models import Job
from model_build.esm_embed import compute_and_save_embeddings, load_all_sequences
from model_build.chemberta_embed import compute_and_save_chem_embeddings, load_all_smiles
from model_build.ppi_classifier import train_classifier
from model_build.ppi_infer import run_inference
from model_build.dti_classifier import train_dti_classifier
from model_build.dti_infer import run_dti_inference
from model_build.rnafm_embed import compute_and_save_rna_embeddings, load_all_rna_sequences
from model_build.rpi_classifier import train_rpi_classifier
from model_build.rpi_infer import run_rpi_inference
from model_build.dnabert_embed import compute_and_save_dna_embeddings, load_all_dna_sequences
from model_build.pdi_classifier import train_pdi_classifier
from model_build.pdi_infer import run_pdi_inference
from email_utils import send_job_notification

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


def _cleanup_run_dir(run_id: str):
    """Delete the artifact directory for a run (embeddings, temp files).
    Called immediately on job failure/cancellation — keeps disk clean."""
    run_dir = os.path.join(MODELS_DIR, run_id)
    if os.path.isdir(run_dir):
        try:
            shutil.rmtree(run_dir)
            print(f"[cleanup] Deleted artifacts for failed run {run_id}", flush=True)
        except Exception as exc:
            print(f"[cleanup] Could not delete {run_dir}: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Task 1 — Training
# ---------------------------------------------------------------------------

@celery.task(name="train_ppi_model")
def train_ppi_model(run_id: str, input_files: list, hyperparams_json: str = "{}"):
    db  = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        if job is None:
            print(f"[{run_id}] ERROR: job row missing — aborting.", flush=True)
            return
        _set_status(db, job, "running")
    except Exception as e:
        print(f"[{run_id}] ERROR setting running status: {e}", flush=True)
        db.close()
        return

    notify_email = ""
    try:
        hyperparams  = json.loads(hyperparams_json)
        notify_email = hyperparams.get("notify_email", "").strip()
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
        send_job_notification(notify_email, run_id, "completed", "ppi", final_metrics)

    except SoftTimeLimitExceeded:
        print(f"[{run_id}] Hit 4-hour time limit — marking failed.", flush=True)
        job.status = "failed"
        job.result = "Job exceeded the 4-hour time limit and was automatically stopped."
        db.commit()
        _cleanup_run_dir(run_id)
        send_job_notification(notify_email, run_id, "failed", "ppi", error_msg="Exceeded 4-hour time limit")

    except Exception as e:
        print(f"[{run_id}] ERROR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        job.status = "failed"
        job.result = str(e)
        db.commit()
        _cleanup_run_dir(run_id)
        send_job_notification(notify_email, run_id, "failed", "ppi", error_msg=str(e))
        raise

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 2 — DTI Training
# ---------------------------------------------------------------------------

@celery.task(name="train_dti_model")
def train_dti_model(run_id: str, input_files: list, hyperparams_json: str = "{}"):
    db  = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        if job is None:
            print(f"[{run_id}] ERROR: job row missing — aborting.", flush=True)
            return
        _set_status(db, job, "running")
    except Exception as e:
        print(f"[{run_id}] ERROR setting running status: {e}", flush=True)
        db.close()
        return

    notify_email = ""
    try:
        hyperparams  = json.loads(hyperparams_json)
        notify_email = hyperparams.get("notify_email", "").strip()
        run_dir      = _run_dir(run_id)
        chem_embed_path = os.path.join(run_dir, f"chem_embedding_{run_id}.pkl")
        esm_embed_path  = os.path.join(run_dir, f"embedding_{run_id}.pkl")
        model_path      = os.path.join(run_dir, f"model_{run_id}.pt")
        metrics_path    = os.path.join(run_dir, f"metrics_{run_id}.json")

        chem_model = hyperparams.get("chem_model", "seyonec/ChemBERTa-zinc-base-v1")
        esm_model  = hyperparams.get("esm_model",  "esm2_t12_35M_UR50D")

        df = pd.concat([pd.read_csv(p) for p in input_files], ignore_index=True)

        # ── Step 1: ChemBERTa SMILES embeddings ──────────────────────────────
        all_smiles = load_all_smiles(input_files, col="smiles")
        print(f"[{run_id}] Unique SMILES: {len(all_smiles)}  model: {chem_model}", flush=True)
        compute_and_save_chem_embeddings(all_smiles, chem_embed_path, chem_model)

        with open(chem_embed_path, "rb") as f:
            chem_dict = pickle.load(f)

        # ── Step 2: ESM2 protein embeddings ──────────────────────────────────
        all_seqs = load_all_sequences(input_files, col_a="sequence", col_b="sequence")
        print(f"[{run_id}] Unique sequences: {len(all_seqs)}  model: {esm_model}", flush=True)
        compute_and_save_embeddings(all_seqs, esm_embed_path, esm_model)

        with open(esm_embed_path, "rb") as f:
            esm_dict = pickle.load(f)

        # ── Step 3: Train classifier ──────────────────────────────────────────
        final_metrics = train_dti_classifier(
            df           = df,
            chem_dict    = chem_dict,
            esm_dict     = esm_dict,
            hyperparams  = hyperparams,
            metrics_path = metrics_path,
            model_path   = model_path,
        )

        job.status     = "completed"
        job.model_path = model_path
        job.metrics    = json.dumps(final_metrics)
        db.commit()
        print(f"[{run_id}] DTI training complete", flush=True)
        send_job_notification(notify_email, run_id, "completed", "dti", final_metrics)

    except SoftTimeLimitExceeded:
        print(f"[{run_id}] Hit 4-hour time limit — marking failed.", flush=True)
        job.status = "failed"
        job.result = "Job exceeded the 4-hour time limit and was automatically stopped."
        db.commit()
        _cleanup_run_dir(run_id)
        send_job_notification(notify_email, run_id, "failed", "dti", error_msg="Exceeded 4-hour time limit")

    except Exception as e:
        print(f"[{run_id}] ERROR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        job.status = "failed"
        job.result = str(e)
        db.commit()
        _cleanup_run_dir(run_id)
        send_job_notification(notify_email, run_id, "failed", "dti", error_msg=str(e))
        raise

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 3 — DTI Inference
# ---------------------------------------------------------------------------

@celery.task(name="run_dti_inference_task")
def run_dti_inference_task(run_id: str, source_run_id: str, input_files: list):
    db  = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        if job is None:
            print(f"[{run_id}] ERROR: job row missing — aborting.", flush=True)
            return
        _set_status(db, job, "running")
    except Exception as e:
        print(f"[{run_id}] ERROR setting running status: {e}", flush=True)
        db.close()
        return

    try:
        run_dir     = _run_dir(run_id)
        results_csv = os.path.join(run_dir, f"results_{run_id}.csv")

        # Fetch source job hyperparams
        src_db = SessionLocal()
        try:
            src_job = src_db.query(Job).filter(Job.run_id == source_run_id).first()
            src_hp  = json.loads(src_job.hyperparams) if src_job and src_job.hyperparams else {}
        finally:
            src_db.close()

        chem_model = src_hp.get("chem_model", "seyonec/ChemBERTa-zinc-base-v1")
        esm_model  = src_hp.get("esm_model",  "esm2_t12_35M_UR50D")

        # Load source training artefacts
        src_dir         = os.path.join(MODELS_DIR, source_run_id)
        chem_embed_path = os.path.join(src_dir, f"chem_embedding_{source_run_id}.pkl")
        esm_embed_path  = os.path.join(src_dir, f"embedding_{source_run_id}.pkl")
        model_path      = os.path.join(src_dir, f"model_{source_run_id}.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        if not os.path.exists(chem_embed_path):
            raise FileNotFoundError(f"Chem embedding file not found: {chem_embed_path}")
        if not os.path.exists(esm_embed_path):
            raise FileNotFoundError(f"Protein embedding file not found: {esm_embed_path}")

        with open(chem_embed_path, "rb") as f:
            chem_dict = pickle.load(f)
        with open(esm_embed_path, "rb") as f:
            esm_dict  = pickle.load(f)

        df = pd.concat([pd.read_csv(p) for p in input_files], ignore_index=True)

        # Embed any SMILES not already cached
        new_smiles = sorted(
            set(df["smiles"].astype(str).str.strip()) - set(chem_dict.keys())
            - {"nan", ""}
        )
        if new_smiles:
            print(
                f"[{run_id}] Computing ChemBERTa embeddings for {len(new_smiles)} new SMILES",
                flush=True,
            )
            tmp_chem_path = os.path.join(run_dir, f"new_chem_{run_id}.pkl")
            compute_and_save_chem_embeddings(new_smiles, tmp_chem_path, chem_model)
            with open(tmp_chem_path, "rb") as f:
                chem_dict.update(pickle.load(f))

        # Embed any sequences not already cached
        new_seqs = sorted(
            set(df["sequence"].astype(str).str.strip().str.upper()) - set(esm_dict.keys())
            - {"NAN", ""}
        )
        if new_seqs:
            print(
                f"[{run_id}] Computing ESM2 embeddings for {len(new_seqs)} new sequences",
                flush=True,
            )
            tmp_esm_path = os.path.join(run_dir, f"new_esm_{run_id}.pkl")
            compute_and_save_embeddings(new_seqs, tmp_esm_path, esm_model)
            with open(tmp_esm_path, "rb") as f:
                esm_dict.update(pickle.load(f))

        results = run_dti_inference(model_path, chem_dict, esm_dict, df)

        results_df = pd.DataFrame(results)
        results_df.to_csv(results_csv, index=False)
        print(f"[{run_id}] DTI inference complete → {results_csv}", flush=True)

        # ── Rich inference metrics ────────────────────────────────────────────
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
        # ─────────────────────────────────────────────────────────────────────

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


# ---------------------------------------------------------------------------
# Task 4 — PPI Inference
# ---------------------------------------------------------------------------

@celery.task(name="run_ppi_inference")
def run_ppi_inference(run_id: str, source_run_id: str, input_files: list):
    db  = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        if job is None:
            print(f"[{run_id}] ERROR: job row missing — aborting.", flush=True)
            return
        _set_status(db, job, "running")
    except Exception as e:
        print(f"[{run_id}] ERROR setting running status: {e}", flush=True)
        db.close()
        return

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


# ---------------------------------------------------------------------------
# Task 5 — RPI Training
# ---------------------------------------------------------------------------

@celery.task(name="train_rpi_model")
def train_rpi_model(run_id: str, input_files: list, hyperparams_json: str = "{}"):
    db  = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        if job is None:
            print(f"[{run_id}] ERROR: job row missing — aborting.", flush=True)
            return
        _set_status(db, job, "running")
    except Exception as e:
        print(f"[{run_id}] ERROR setting running status: {e}", flush=True)
        db.close()
        return

    notify_email = ""
    try:
        hyperparams    = json.loads(hyperparams_json)
        notify_email   = hyperparams.get("notify_email", "").strip()
        run_dir        = _run_dir(run_id)
        rna_embed_path = os.path.join(run_dir, f"rna_embedding_{run_id}.pkl")
        esm_embed_path = os.path.join(run_dir, f"embedding_{run_id}.pkl")
        model_path     = os.path.join(run_dir, f"model_{run_id}.pt")
        metrics_path   = os.path.join(run_dir, f"metrics_{run_id}.json")

        rna_model = hyperparams.get("rna_model", "multimolecule/rnafm")
        esm_model = hyperparams.get("esm_model", "esm2_t12_35M_UR50D")

        df = pd.concat([pd.read_csv(p) for p in input_files], ignore_index=True)

        # Step 1: RNA-FM embeddings
        all_rna = load_all_rna_sequences(input_files, col="rna_sequence")
        print(f"[{run_id}] Unique RNA sequences: {len(all_rna)}  model: {rna_model}", flush=True)
        compute_and_save_rna_embeddings(all_rna, rna_embed_path, rna_model)

        with open(rna_embed_path, "rb") as f:
            rna_dict = pickle.load(f)

        # Step 2: ESM2 protein embeddings
        all_seqs = load_all_sequences(input_files, col_a="protein_sequence", col_b="protein_sequence")
        print(f"[{run_id}] Unique protein sequences: {len(all_seqs)}  model: {esm_model}", flush=True)
        compute_and_save_embeddings(all_seqs, esm_embed_path, esm_model)

        with open(esm_embed_path, "rb") as f:
            esm_dict = pickle.load(f)

        # Step 3: Train classifier
        final_metrics = train_rpi_classifier(
            df           = df,
            rna_dict     = rna_dict,
            esm_dict     = esm_dict,
            hyperparams  = hyperparams,
            metrics_path = metrics_path,
            model_path   = model_path,
        )

        job.status     = "completed"
        job.model_path = model_path
        job.metrics    = json.dumps(final_metrics)
        db.commit()
        print(f"[{run_id}] RPI training complete", flush=True)
        send_job_notification(notify_email, run_id, "completed", "rpi", final_metrics)

    except SoftTimeLimitExceeded:
        print(f"[{run_id}] Hit 4-hour time limit — marking failed.", flush=True)
        job.status = "failed"
        job.result = "Job exceeded the 4-hour time limit and was automatically stopped."
        db.commit()
        _cleanup_run_dir(run_id)
        send_job_notification(notify_email, run_id, "failed", "rpi", error_msg="Exceeded 4-hour time limit")

    except Exception as e:
        print(f"[{run_id}] ERROR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        job.status = "failed"
        job.result = str(e)
        db.commit()
        _cleanup_run_dir(run_id)
        send_job_notification(notify_email, run_id, "failed", "rpi", error_msg=str(e))
        raise

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 6 — RPI Inference
# ---------------------------------------------------------------------------

@celery.task(name="run_rpi_inference_task")
def run_rpi_inference_task(run_id: str, source_run_id: str, input_files: list):
    db  = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        if job is None:
            print(f"[{run_id}] ERROR: job row missing — aborting.", flush=True)
            return
        _set_status(db, job, "running")
    except Exception as e:
        print(f"[{run_id}] ERROR setting running status: {e}", flush=True)
        db.close()
        return

    try:
        run_dir     = _run_dir(run_id)
        results_csv = os.path.join(run_dir, f"results_{run_id}.csv")

        src_db = SessionLocal()
        try:
            src_job = src_db.query(Job).filter(Job.run_id == source_run_id).first()
            src_hp  = json.loads(src_job.hyperparams) if src_job and src_job.hyperparams else {}
        finally:
            src_db.close()

        rna_model = src_hp.get("rna_model", "multimolecule/rnafm")
        esm_model = src_hp.get("esm_model", "esm2_t12_35M_UR50D")

        src_dir        = os.path.join(MODELS_DIR, source_run_id)
        rna_embed_path = os.path.join(src_dir, f"rna_embedding_{source_run_id}.pkl")
        esm_embed_path = os.path.join(src_dir, f"embedding_{source_run_id}.pkl")
        model_path     = os.path.join(src_dir, f"model_{source_run_id}.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        if not os.path.exists(rna_embed_path):
            raise FileNotFoundError(f"RNA embedding file not found: {rna_embed_path}")
        if not os.path.exists(esm_embed_path):
            raise FileNotFoundError(f"Protein embedding file not found: {esm_embed_path}")

        with open(rna_embed_path, "rb") as f:
            rna_dict = pickle.load(f)
        with open(esm_embed_path, "rb") as f:
            esm_dict = pickle.load(f)

        df = pd.concat([pd.read_csv(p) for p in input_files], ignore_index=True)

        # Embed any RNA not already cached
        new_rna = sorted(
            set(
                df["rna_sequence"].astype(str).str.strip().str.upper()
                .str.replace("T", "U", regex=False)
            ) - set(rna_dict.keys()) - {"NAN", ""}
        )
        if new_rna:
            print(f"[{run_id}] Computing RNA-FM embeddings for {len(new_rna)} new sequences", flush=True)
            tmp_rna_path = os.path.join(run_dir, f"new_rna_{run_id}.pkl")
            compute_and_save_rna_embeddings(new_rna, tmp_rna_path, rna_model)
            with open(tmp_rna_path, "rb") as f:
                rna_dict.update(pickle.load(f))

        # Embed any protein sequences not already cached
        new_seqs = sorted(
            set(df["protein_sequence"].astype(str).str.strip().str.upper())
            - set(esm_dict.keys()) - {"NAN", ""}
        )
        if new_seqs:
            print(f"[{run_id}] Computing ESM2 embeddings for {len(new_seqs)} new sequences", flush=True)
            tmp_esm_path = os.path.join(run_dir, f"new_esm_{run_id}.pkl")
            compute_and_save_embeddings(new_seqs, tmp_esm_path, esm_model)
            with open(tmp_esm_path, "rb") as f:
                esm_dict.update(pickle.load(f))

        results = run_rpi_inference(model_path, rna_dict, esm_dict, df)

        results_df = pd.DataFrame(results)
        results_df.to_csv(results_csv, index=False)
        print(f"[{run_id}] RPI inference complete → {results_csv}", flush=True)

        probs_list = [r["probability"] for r in results if r.get("probability") is not None]
        has_labels = "label" in df.columns
        infer_metrics: dict = {"has_labels": has_labels, "probabilities": probs_list}

        if has_labels and probs_list:
            import math as _math
            from sklearn.metrics import (
                roc_auc_score, average_precision_score,
                f1_score, accuracy_score, matthews_corrcoef,
            )
            valid_idx = [i for i, r in enumerate(results) if r.get("probability") is not None]
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


# ---------------------------------------------------------------------------
# Task 7 — PDI Training
# ---------------------------------------------------------------------------

@celery.task(name="train_pdi_model")
def train_pdi_model(run_id: str, input_files: list, hyperparams_json: str = "{}"):
    db  = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        if job is None:
            print(f"[{run_id}] ERROR: job row missing — aborting.", flush=True)
            return
        _set_status(db, job, "running")
    except Exception as e:
        print(f"[{run_id}] ERROR setting running status: {e}", flush=True)
        db.close()
        return

    notify_email = ""
    try:
        hyperparams    = json.loads(hyperparams_json)
        notify_email   = hyperparams.get("notify_email", "").strip()
        run_dir        = _run_dir(run_id)
        dna_embed_path = os.path.join(run_dir, f"dna_embedding_{run_id}.pkl")
        esm_embed_path = os.path.join(run_dir, f"embedding_{run_id}.pkl")
        model_path     = os.path.join(run_dir, f"model_{run_id}.pt")
        metrics_path   = os.path.join(run_dir, f"metrics_{run_id}.json")

        dna_model = hyperparams.get("dna_model", "armheb/DNA_bert_6")
        esm_model = hyperparams.get("esm_model", "esm2_t12_35M_UR50D")

        df = pd.concat([pd.read_csv(p) for p in input_files], ignore_index=True)

        # Step 1: DNABERT-2 embeddings
        all_dna = load_all_dna_sequences(input_files, col="dna_sequence")
        print(f"[{run_id}] Unique DNA sequences: {len(all_dna)}  model: {dna_model}", flush=True)
        compute_and_save_dna_embeddings(all_dna, dna_embed_path, dna_model)

        with open(dna_embed_path, "rb") as f:
            dna_dict = pickle.load(f)

        # Step 2: ESM2 protein embeddings
        all_seqs = load_all_sequences(input_files, col_a="protein_sequence", col_b="protein_sequence")
        print(f"[{run_id}] Unique protein sequences: {len(all_seqs)}  model: {esm_model}", flush=True)
        compute_and_save_embeddings(all_seqs, esm_embed_path, esm_model)

        with open(esm_embed_path, "rb") as f:
            esm_dict = pickle.load(f)

        # Step 3: Train classifier
        final_metrics = train_pdi_classifier(
            df           = df,
            dna_dict     = dna_dict,
            esm_dict     = esm_dict,
            hyperparams  = hyperparams,
            metrics_path = metrics_path,
            model_path   = model_path,
        )

        job.status     = "completed"
        job.model_path = model_path
        job.metrics    = json.dumps(final_metrics)
        db.commit()
        print(f"[{run_id}] PDI training complete", flush=True)
        send_job_notification(notify_email, run_id, "completed", "pdi", final_metrics)

    except SoftTimeLimitExceeded:
        print(f"[{run_id}] Hit 4-hour time limit — marking failed.", flush=True)
        job.status = "failed"
        job.result = "Job exceeded the 4-hour time limit and was automatically stopped."
        db.commit()
        _cleanup_run_dir(run_id)
        send_job_notification(notify_email, run_id, "failed", "pdi", error_msg="Exceeded 4-hour time limit")

    except Exception as e:
        print(f"[{run_id}] ERROR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        job.status = "failed"
        job.result = str(e)
        db.commit()
        _cleanup_run_dir(run_id)
        send_job_notification(notify_email, run_id, "failed", "pdi", error_msg=str(e))
        raise

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 8 — PDI Inference
# ---------------------------------------------------------------------------

@celery.task(name="run_pdi_inference_task")
def run_pdi_inference_task(run_id: str, source_run_id: str, input_files: list):
    db  = SessionLocal()
    try:
        job = db.query(Job).filter(Job.run_id == run_id).first()
        if job is None:
            print(f"[{run_id}] ERROR: job row missing — aborting.", flush=True)
            return
        _set_status(db, job, "running")
    except Exception as e:
        print(f"[{run_id}] ERROR setting running status: {e}", flush=True)
        db.close()
        return

    try:
        run_dir     = _run_dir(run_id)
        results_csv = os.path.join(run_dir, f"results_{run_id}.csv")

        src_db = SessionLocal()
        try:
            src_job = src_db.query(Job).filter(Job.run_id == source_run_id).first()
            src_hp  = json.loads(src_job.hyperparams) if src_job and src_job.hyperparams else {}
        finally:
            src_db.close()

        dna_model = src_hp.get("dna_model", "multimolecule/dnabert2")
        esm_model = src_hp.get("esm_model", "esm2_t12_35M_UR50D")

        src_dir        = os.path.join(MODELS_DIR, source_run_id)
        dna_embed_path = os.path.join(src_dir, f"dna_embedding_{source_run_id}.pkl")
        esm_embed_path = os.path.join(src_dir, f"embedding_{source_run_id}.pkl")
        model_path     = os.path.join(src_dir, f"model_{source_run_id}.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        if not os.path.exists(dna_embed_path):
            raise FileNotFoundError(f"DNA embedding file not found: {dna_embed_path}")
        if not os.path.exists(esm_embed_path):
            raise FileNotFoundError(f"Protein embedding file not found: {esm_embed_path}")

        with open(dna_embed_path, "rb") as f:
            dna_dict = pickle.load(f)
        with open(esm_embed_path, "rb") as f:
            esm_dict = pickle.load(f)

        df = pd.concat([pd.read_csv(p) for p in input_files], ignore_index=True)

        # Embed any DNA sequences not already cached
        new_dna = sorted(
            set(df["dna_sequence"].astype(str).str.strip().str.upper())
            - set(dna_dict.keys()) - {"NAN", ""}
        )
        if new_dna:
            print(f"[{run_id}] Computing DNABERT-2 embeddings for {len(new_dna)} new sequences", flush=True)
            tmp_dna_path = os.path.join(run_dir, f"new_dna_{run_id}.pkl")
            compute_and_save_dna_embeddings(new_dna, tmp_dna_path, dna_model)
            with open(tmp_dna_path, "rb") as f:
                dna_dict.update(pickle.load(f))

        # Embed any protein sequences not already cached
        new_seqs = sorted(
            set(df["protein_sequence"].astype(str).str.strip().str.upper())
            - set(esm_dict.keys()) - {"NAN", ""}
        )
        if new_seqs:
            print(f"[{run_id}] Computing ESM2 embeddings for {len(new_seqs)} new sequences", flush=True)
            tmp_esm_path = os.path.join(run_dir, f"new_esm_{run_id}.pkl")
            compute_and_save_embeddings(new_seqs, tmp_esm_path, esm_model)
            with open(tmp_esm_path, "rb") as f:
                esm_dict.update(pickle.load(f))

        results = run_pdi_inference(model_path, dna_dict, esm_dict, df)

        results_df = pd.DataFrame(results)
        results_df.to_csv(results_csv, index=False)
        print(f"[{run_id}] PDI inference complete → {results_csv}", flush=True)

        probs_list = [r["probability"] for r in results if r.get("probability") is not None]
        has_labels = "label" in df.columns
        infer_metrics: dict = {"has_labels": has_labels, "probabilities": probs_list}

        if has_labels and probs_list:
            import math as _math
            from sklearn.metrics import (
                roc_auc_score, average_precision_score,
                f1_score, accuracy_score, matthews_corrcoef,
            )
            valid_idx = [i for i, r in enumerate(results) if r.get("probability") is not None]
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
