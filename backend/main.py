
## Backend file - main.py
###################################
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from typing import List
from pydantic import BaseModel
import pandas as pd
import io
import pickle
import json
import os




MODELS_DIR = "/app/saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def create_run_folder(run_id: str):
    run_dir = os.path.join(MODELS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
###################################

from fastapi.middleware.cors import CORSMiddleware
import uuid
from celery.result import AsyncResult

##### import from other files ####################
from tasks import train_ppi_model
from database import Base, engine, SessionLocal
from models import Job



##################################################
app = FastAPI()
##################################################

@app.get("/health")
def health():
    return {"status": "ok"}

# # Create tables automatically
Base.metadata.create_all(bind=engine)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/create_job")
async def create_job(files: List[UploadFile] = File(...)):
    run_id = str(uuid.uuid4())[:8]
    run_dir = create_run_folder(run_id)

    input_files = []

    for file in files:
        contents = await file.read()
        path = f"{run_dir}/{run_id}_{file.filename}"

        with open(path, "wb") as f:
            f.write(contents)

        input_files.append(path)

    # save list of CSV paths
    db = SessionLocal()
    job = Job(
        run_id=run_id,
        status="queued",
        input_sequence=json.dumps(input_files)
    )
    db.add(job)
    db.commit()
    db.close()

    # send list of file paths to celery
    train_ppi_model.delay(run_id, input_files)

    return {"run_id": run_id}




@app.get("/check_status/{run_id}")
def check_status(run_id: str):
    db = SessionLocal()
    job = db.query(Job).filter(Job.run_id == run_id).first()

    if not job:
        return {"error": "invalid id"}

    return {
        "run_id": job.run_id,
        "status": job.status,
        "result": job.result
    }



@app.get("/download_embedding/{run_id}")
def download_embedding(run_id: str):
    run_dir = os.path.join(MODELS_DIR, run_id)
    embed_path = os.path.join(run_dir, f"embedding_{run_id}.pkl")

    if not os.path.exists(embed_path):
        return {"error": "embedding not found"}

    return FileResponse(
        embed_path,
        media_type="application/octet-stream",
        filename=f"embedding_{run_id}.pkl"
    )
















