#!/usr/bin/env python3
"""Promote a completed training run into the permanent example-run registry."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


TASK_TO_EXAMPLE_ID = {
    "ppi": "example_ppi",
    "dtpi": "example_dtpi",
    "rpi": "example_rpi",
    "pdi": "example_pdi",
}

TASK_TO_TITLE = {
    "ppi": "Example PPI Model",
    "dtpi": "Example DTPI Model",
    "rpi": "Example RPI Model",
    "pdi": "Example PDI Model",
}

DEFAULT_BACKEND_DIR = Path(__file__).resolve().parents[1]


def _load_config(run_dir: Path, run_id: str) -> dict:
    zip_path = run_dir / f"artifacts_{run_id}.zip"
    config_name = f"config_{run_id}.json"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing artifact bundle: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(config_name) as fh:
            return json.load(fh)


def _copy_json(run_dir: Path, run_id: str, src_name: str, out_dir: Path, dst_name: str) -> None:
    src = run_dir / src_name.format(run_id=run_id)
    if not src.exists():
        raise FileNotFoundError(f"Missing required example artifact: {src}")
    shutil.copy2(src, out_dir / dst_name)


def promote(task: str, run_id: str, saved_models: Path, example_runs: Path) -> None:
    if task not in TASK_TO_EXAMPLE_ID:
        raise ValueError(f"Unsupported task {task!r}; expected one of {sorted(TASK_TO_EXAMPLE_ID)}")

    run_dir = saved_models / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out_dir = example_runs / task
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(run_dir, run_id)
    metadata = {
        "run_id": TASK_TO_EXAMPLE_ID[task],
        "source_run_id": run_id,
        "status": "completed",
        "job_type": "train",
        "task_type": task,
        "title": TASK_TO_TITLE[task],
        "result": None,
        "hyperparams": config.get("hyperparams", {}),
    }

    _copy_json(run_dir, run_id, "metrics_{run_id}.json", out_dir, "metrics.json")
    _copy_json(run_dir, run_id, "dataset_stats_{run_id}.json", out_dir, "dataset_stats.json")
    _copy_json(run_dir, run_id, "emb_umap_{run_id}.json", out_dir, "emb_umap.json")
    _copy_json(run_dir, run_id, "model_umap_{run_id}.json", out_dir, "model_umap.json")

    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")

    print(f"Promoted {task.upper()} run {run_id} -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASK_TO_EXAMPLE_ID))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--saved-models", default=str(DEFAULT_BACKEND_DIR / "saved_models"))
    parser.add_argument("--example-runs", default=str(DEFAULT_BACKEND_DIR / "example_runs"))
    args = parser.parse_args()

    promote(
        task=args.task,
        run_id=args.run_id,
        saved_models=Path(args.saved_models),
        example_runs=Path(args.example_runs),
    )


if __name__ == "__main__":
    main()
