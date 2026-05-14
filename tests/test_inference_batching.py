import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from model_build.chemberta_embed import load_all_smiles
from model_build.dnabert_embed import load_all_dna_sequences
from model_build.dtpi_infer import run_dtpi_inference
import model_build.dtpi_infer as dtpi_infer
from model_build.pdi_infer import run_pdi_inference
import model_build.pdi_infer as pdi_infer
from model_build.ppi_classifier import FlexiblePPIModel
from model_build.ppi_infer import run_inference
import model_build.ppi_infer as ppi_infer
from model_build.rnafm_embed import load_all_rna_sequences
from model_build.rpi_infer import run_rpi_inference
import model_build.rpi_infer as rpi_infer
from model_build.esm_embed import load_all_sequences


def _save_pooled_checkpoint(path, input_dim, extra=None):
    layer_configs = [
        {"type": "linear", "hidden_dim": 4, "activation": "relu", "dropout": 0.0}
    ]
    model = FlexiblePPIModel(input_dim, layer_configs)
    ckpt = {
        "model_state": model.state_dict(),
        "input_dim": input_dim,
        "layer_configs": layer_configs,
        "embedding_representation": "pooled",
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def _fake_score(unique_pairs, *_args, **_kwargs):
    assert unique_pairs == [("A", "B")]
    return {("A", "B"): 0.75}


def test_ppi_inference_scores_duplicate_pair_once(tmp_path, monkeypatch):
    model_path = tmp_path / "ppi.pt"
    _save_pooled_checkpoint(model_path, 4)
    monkeypatch.setattr(ppi_infer, "score_pooled_pairs", _fake_score)
    embeddings = {"A": torch.ones(2), "B": torch.ones(2)}
    df = pd.DataFrame({
        "proteinA": [" a ", "A", "missing"],
        "proteinB": [" b ", "B", "B"],
    })

    results = run_inference(str(model_path), embeddings, df)

    assert [row["probability"] for row in results] == [0.75, 0.75, None]
    assert results[2]["note"] == "embedding not available"


@pytest.mark.parametrize(
    ("runner", "module", "left_key", "right_key", "left_label", "right_label", "extra"),
    [
        (run_dtpi_inference, dtpi_infer, "smiles", "sequence", "compound", "protein", {"chem_dim": 2, "esm_dim": 2, "task_type": "dtpi"}),
        (run_rpi_inference, rpi_infer, "rna_sequence", "protein_sequence", "RNA", "protein", {"rna_dim": 2, "esm_dim": 2, "task_type": "rpi"}),
        (run_pdi_inference, pdi_infer, "dna_sequence", "protein_sequence", "DNA", "protein", {"dna_dim": 2, "esm_dim": 2, "task_type": "pdi"}),
    ],
)
def test_two_sided_inference_scores_duplicate_pair_once(
    tmp_path, monkeypatch, runner, module, left_key, right_key, left_label, right_label, extra
):
    model_path = tmp_path / f"{extra['task_type']}.pt"
    _save_pooled_checkpoint(model_path, 4, extra)
    monkeypatch.setattr(module, "score_pooled_pairs", _fake_score)
    left_dict = {"A": torch.ones(2)}
    right_dict = {"B": torch.ones(2)}
    df = pd.DataFrame({
        left_key: ["A", "A", "missing"],
        right_key: ["B", "B", "B"],
    })

    results = runner(str(model_path), left_dict, right_dict, df)

    assert [row["probability"] for row in results] == [0.75, 0.75, None]
    assert left_label in results[2]["note"]
    assert right_label not in results[2]["note"]


def test_training_embedding_loaders_return_unique_inputs(tmp_path):
    ppi = tmp_path / "ppi.csv"
    ppi.write_text("proteinA,proteinB,label\nAAA,BBB,1\naaa,BBB,0\n", encoding="utf-8")
    dtpi = tmp_path / "dtpi.csv"
    dtpi.write_text("smiles,sequence,label\nCCO,AAA,1\nCCO,aaa,0\n", encoding="utf-8")
    rpi = tmp_path / "rpi.csv"
    rpi.write_text("rna_sequence,protein_sequence,label\nAUG,AAA,1\nATG,AAA,0\n", encoding="utf-8")
    pdi = tmp_path / "pdi.csv"
    pdi.write_text("dna_sequence,protein_sequence,label\nATGC,AAA,1\natgc,AAA,0\n", encoding="utf-8")

    assert load_all_sequences([ppi]) == ["AAA", "BBB"]
    assert load_all_smiles([dtpi], col="smiles") == ["CCO"]
    assert load_all_sequences([dtpi], col_a="sequence", col_b="sequence") == ["AAA"]
    assert load_all_rna_sequences([rpi], col="rna_sequence") == ["AUG"]
    assert load_all_dna_sequences([pdi], col="dna_sequence") == ["ATGC"]
    assert load_all_sequences([pdi], col_a="protein_sequence", col_b="protein_sequence") == ["AAA"]
