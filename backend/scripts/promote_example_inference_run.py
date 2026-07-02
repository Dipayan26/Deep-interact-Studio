#!/usr/bin/env python3
"""Promote a completed inference run into the permanent example-inference registry."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


TASK_TO_EXAMPLE_ID = {
    "ppi": "example_infer_ppi",
    "dtpi": "example_infer_dtpi",
    "rpi": "example_infer_rpi",
    "pdi": "example_infer_pdi",
}

TASK_TO_TITLE = {
    "ppi": "Example PPI Inference",
    "dtpi": "Example DTPI Inference",
    "rpi": "Example RPI Inference",
    "pdi": "Example PDI Inference",
}

DEFAULT_BACKEND_DIR = Path(__file__).resolve().parents[1]


def _load_zip_config(run_dir: Path, run_id: str) -> dict | None:
    zip_path = run_dir / f"artifacts_{run_id}.zip"
    config_name = f"config_{run_id}.json"
    if not zip_path.exists():
        return None
    with zipfile.ZipFile(zip_path) as zf:
        if config_name not in zf.namelist():
            return None
        with zf.open(config_name) as fh:
            return json.load(fh)


def _load_source_config(saved_models: Path, source_run_id: str) -> dict:
    source_dir = saved_models / source_run_id
    config = _load_zip_config(source_dir, source_run_id)
    if config is not None:
        return config
    return {
        "run_id": source_run_id,
        "status": "completed",
        "job_type": "train",
        "source_run_id": None,
        "hyperparams": {},
        "result": None,
    }


def _copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required example artifact: {src}")
    shutil.copy2(src, dst)


def promote(
    task: str,
    run_id: str,
    source_run_id: str,
    saved_models: Path,
    example_runs: Path,
    shap_path: Path | None,
) -> None:
    if task not in TASK_TO_EXAMPLE_ID:
        raise ValueError(f"Unsupported task {task!r}; expected one of {sorted(TASK_TO_EXAMPLE_ID)}")

    run_dir = saved_models / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Inference run directory not found: {run_dir}")

    out_dir = example_runs / task
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _load_zip_config(run_dir, run_id) or {
        "run_id": run_id,
        "status": "completed",
        "job_type": "inference",
        "source_run_id": source_run_id,
        "hyperparams": {
            "task_type": task,
            "infer_label": "",
            "is_single": False,
        },
        "result": str(run_dir / f"results_{run_id}.csv"),
    }
    hyperparams = config.get("hyperparams", {}) or {}
    hyperparams["task_type"] = task
    hyperparams.setdefault("is_single", False)

    metadata = {
        "run_id": TASK_TO_EXAMPLE_ID[task],
        "source_run_id": source_run_id,
        "source_inference_run_id": run_id,
        "status": "completed",
        "job_type": "inference",
        "task_type": task,
        "title": TASK_TO_TITLE[task],
        "result": "results.csv",
        "hyperparams": hyperparams,
    }

    source_config = _load_source_config(saved_models, source_run_id)
    source_metadata = {
        "run_id": source_run_id,
        "status": source_config.get("status", "completed"),
        "job_type": "train",
        "hyperparams": source_config.get("hyperparams", {}),
        "source_run_id": None,
    }

    _copy_required(run_dir / f"results_{run_id}.csv", out_dir / "results.csv")
    _copy_required(run_dir / f"infer_metrics_{run_id}.json", out_dir / "inference_metrics.json")
    _copy_required(saved_models / source_run_id / f"metrics_{source_run_id}.json", out_dir / "source_metrics.json")

    if shap_path is not None and shap_path.exists():
        _copy_required(shap_path, out_dir / "shap.json")
    else:
        existing_shap = run_dir / f"shap_{run_id}.json"
        if existing_shap.exists():
            _copy_required(existing_shap, out_dir / "shap.json")

    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")
    with open(out_dir / "source_metadata.json", "w") as fh:
        json.dump(source_metadata, fh, indent=2)
        fh.write("\n")

    print(f"Promoted {task.upper()} inference run {run_id} -> {out_dir}")
    if not (out_dir / "shap.json").exists():
        print(f"SHAP cache missing for {task.upper()}; add {out_dir / 'shap.json'} when computed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASK_TO_EXAMPLE_ID))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--saved-models", default=str(DEFAULT_BACKEND_DIR / "saved_models"))
    parser.add_argument("--example-runs", default=str(DEFAULT_BACKEND_DIR / "example_inference_runs"))
    parser.add_argument("--shap-path")
    args = parser.parse_args()

    promote(
        task=args.task,
        run_id=args.run_id,
        source_run_id=args.source_run_id,
        saved_models=Path(args.saved_models),
        example_runs=Path(args.example_runs),
        shap_path=Path(args.shap_path) if args.shap_path else None,
    )


if __name__ == "__main__":
    main()
