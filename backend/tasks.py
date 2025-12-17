from celery import Celery
import torch
import torch.nn as nn
import json
import os

from database import SessionLocal
from models import Job
import traceback
################################################
from model_build.esm_embed import compute_and_save_embeddings, load_all_sequences
################################################

########################################################
MODELS_DIR = "/app/saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def create_run_folder(run_id: str):
    run_dir = os.path.join(MODELS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


########################################################

celery = Celery(
    "ppi_tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/1",
)


# ---- PyTorch Model  ----
# def tokenize(seq):
#     mapping = {c: i+1 for i, c in enumerate("ACGTBDEFHIJKLMNOPQRSTUVWXYZ")}
#     return [mapping.get(c.upper(), 0) for c in seq][:150]


######## model 
class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(30, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])




@celery.task(name="train_ppi_model")
def train_ppi_model(run_id, input_files):
    """
    input_files: List[str]  # CSV file paths
    """

    db = SessionLocal()
    job = db.query(Job).filter(Job.run_id == run_id).first()
    job.status = "running"
    db.commit()

    try:
            print("Task started", run_id, input_files, flush=True)

            run_dir = create_run_folder(run_id)
            print("Run dir:", run_dir, flush=True)

            embedding_path = os.path.join(run_dir, f"embedding_{run_id}.pkl")
            print("Embedding path:", embedding_path, flush=True)

            seqs = load_all_sequences(input_files)
            print("Loaded sequences:", len(seqs), flush=True)

            compute_and_save_embeddings(
                all_sequences=seqs,
                outfile=embedding_path
            )
            print("Embedding saved", flush=True)

            # Update database
            job.status = "completed"
            job.result = json.dumps({
                "prediction": 1,
                "probability": 0.8
            })
            db.commit()

    except Exception as e:
        print("ERROR:", str(e), flush=True)
        print(traceback.format_exc(), flush=True)
        raise e


