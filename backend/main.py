
## Backend file - main.py
###################################
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import io
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid, threading, time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------
# FASTAPI SETUP
# -------------------------------------------------------
# app = FastAPI()
JOBS = {}   # In-memory database for demo

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# DEEP LEARNING CODE
# -------------------------------------------------------

# Simple tokenizer (A,T,G,C or any characters)
def tokenize(sequence):
    mapping = {c: i+1 for i, c in enumerate("ACGTBDEFHIJKLMNOPQRSTUVWXYZ")}
    return [mapping.get(c.upper(), 0) for c in sequence][:200]

class SequenceDataset(Dataset):
    def __init__(self, seq):
        self.x = torch.tensor([tokenize(seq)], dtype=torch.long)
        self.y = torch.tensor([1], dtype=torch.long)  # dummy label

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SimpleLSTM(nn.Module):
    def __init__(self, vocab=30, embed_dim=16, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

def train_model(seq):
    dataset = SequenceDataset(seq)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = SimpleLSTM()
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(8):  # small training loop
        for x, y in loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

    # prediction
    with torch.no_grad():
        logits = model(dataset.x)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs).item()

    return {
        "prediction": int(pred_class),
        "probability": float(probs[0][pred_class]),
    }

# -------------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------------

class JobRequest(BaseModel):
    sequence: str

def background_training(run_id, seq):
    JOBS[run_id]["status"] = "running"

    results = train_model(seq)

    JOBS[run_id]["status"] = "completed"
    JOBS[run_id]["results"] = results

@app.post("/create_job")
def create_job(req: JobRequest):

    run_id = str(uuid.uuid4())[:8]

    JOBS[run_id] = {
        "status": "queued",
        "sequence": req.sequence,
        "results": None,
    }

    # start training in background
    thread = threading.Thread(target=background_training, args=(run_id, req.sequence))
    thread.start()

    return {"run_id": run_id, "message": "Training started"}

@app.get("/check_status/{run_id}")
def check_status(run_id: str):
    if run_id not in JOBS:
        return {"error": "Invalid run id"}

    return JOBS[run_id]
