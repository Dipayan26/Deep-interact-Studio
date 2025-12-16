
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


from fastapi.middleware.cors import CORSMiddleware
import uuid
from celery.result import AsyncResult

##### import from other files ####################
from tasks import train_ppi_model
from database import Base, engine, SessionLocal
from models import Job
##################################################

# Create tables automatically
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobRequest(BaseModel):
    sequence: str

@app.post("/create_job")
def create_job(req: JobRequest):
    run_id = str(uuid.uuid4())[:8]

    db = SessionLocal()
    job = Job(
        run_id=run_id,
        status="queued",
        input_sequence=req.sequence
    )
    db.add(job)
    db.commit()

    # queue task
    task = train_ppi_model.delay(run_id, req.sequence)

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



















