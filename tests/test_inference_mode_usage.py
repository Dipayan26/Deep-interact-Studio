from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TARGETED_BACKEND_PATHS = [
    ROOT / "backend" / "main.py",
    ROOT / "backend" / "model_build" / "ppi_infer.py",
    ROOT / "backend" / "model_build" / "dtpi_infer.py",
    ROOT / "backend" / "model_build" / "rpi_infer.py",
    ROOT / "backend" / "model_build" / "pdi_infer.py",
    ROOT / "backend" / "model_build" / "inference_batching.py",
    ROOT / "backend" / "model_build" / "ppi_classifier.py",
    ROOT / "backend" / "model_build" / "dtpi_classifier.py",
    ROOT / "backend" / "model_build" / "rpi_classifier.py",
    ROOT / "backend" / "model_build" / "pdi_classifier.py",
    ROOT / "backend" / "model_build" / "chunked_pair_classifier.py",
]


def test_targeted_inference_paths_use_inference_mode_instead_of_no_grad():
    offenders = []
    for path in TARGETED_BACKEND_PATHS:
        text = path.read_text()
        if "torch.no_grad(" in text:
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []

