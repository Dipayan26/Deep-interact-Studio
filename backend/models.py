from sqlalchemy import Column, String, Text, DateTime
from database import Base
import datetime


class Job(Base):
    __tablename__ = "jobs"

    run_id        = Column(String, primary_key=True, index=True)
    status        = Column(String,  default="queued")
    job_type      = Column(String,  default="train")   # "train" | "inference"
    input_sequence = Column(Text)                       # JSON list of input CSV paths
    hyperparams   = Column(Text)                        # JSON training hyperparams
    model_path    = Column(String)                      # path to saved .pt file
    metrics       = Column(Text)                        # JSON metrics (written per-epoch)
    source_run_id     = Column(String)   # inference jobs: training run_id
    cancel_token_hash = Column(Text)     # SHA-256 of the one-time cancel token
    celery_task_id    = Column(String)   # Celery AsyncResult.id for revocation
    result            = Column(Text)     # legacy / misc
    created_at        = Column(DateTime, default=datetime.datetime.utcnow)
