from sqlalchemy import Column, String, Text, DateTime
from database import Base
import datetime

class Job(Base):
    __tablename__ = "jobs"

    run_id = Column(String, primary_key=True, index=True)
    status = Column(String, default="queued")
    input_sequence = Column(Text)
    result = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)



# class Job(Base):
#     __tablename__ = "jobs"

#     run_id = Column(String, primary_key=True)
#     status = Column(String, default="queued")

#     # File paths instead of actual files
#     model_path = Column(String)
#     logs_path = Column(String)
#     train_plot_path = Column(String)
#     test_plot_path = Column(String)

#     # Store summary results (small)
#     metrics = Column(Text)  # JSON string
#     created_at = Column(DateTime, default=datetime.datetime.utcnow)
