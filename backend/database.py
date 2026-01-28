from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./wagon_inspection.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    video_path = Column(String, index=True)
    direction = Column(String, default="LR")
    camera_angle = Column(String, default="LR")  # NEW: LR, RL, or TOP
    status = Column(String, default="created")  # created, processing, completed, failed
    progress = Column(Float, default=0.0)
    current_stage = Column(String, default="Initialized")
    
    created_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    
    # Store paths to result files
    output_dir = Column(String, nullable=True)
    result_json_path = Column(String, nullable=True)
    
    # Simple summary stats for the dashboard list view
    total_wagons = Column(Integer, default=0)
    defects_count = Column(Integer, default=0)
    
    # Live counts during processing (updated in real-time)
    live_counts = Column(JSON, default={})
    
    error_message = Column(Text, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()