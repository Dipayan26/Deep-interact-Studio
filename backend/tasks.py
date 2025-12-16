from celery import Celery
import torch
import torch.nn as nn
import json
import os

from database import SessionLocal
from models import Job

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

# ---- PyTorch Model (same as before) ----
def tokenize(seq):
    mapping = {c: i+1 for i, c in enumerate("ACGTBDEFHIJKLMNOPQRSTUVWXYZ")}
    return [mapping.get(c.upper(), 0) for c in seq][:150]

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
def train_ppi_model(run_id, sequence):
    db = SessionLocal()

    job = db.query(Job).filter(Job.run_id == run_id).first()
    job.status = "running"
    db.commit()

    # Train the model (like before)
    x = torch.tensor([tokenize(sequence)], dtype=torch.long)
    y = torch.tensor([1])

    model = SimpleLSTM()
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(8):
        pred = model(x)
        loss = loss_fn(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    probs = torch.softmax(model(x), dim=1)
    pred_class = torch.argmax(probs).item()
    #################
    run_dir = create_run_folder(run_id)
    model_path = os.path.join(run_dir, f"model_{run_id}.pt")
    # torch.save(model.state_dict(), f"backend//{run_id}/model.pt")
    torch.save(model.state_dict(), model_path)
    
    #################
    job.status = "completed"
    job.result = json.dumps({
        "prediction": int(pred_class),
        "probability": float(probs[0][pred_class])
    })

    db.commit()
    return "done"
