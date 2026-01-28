from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import shutil
from pathlib import Path
from datetime import datetime

from database import get_db, Job
from schemas import JobCreate, JobResponse, JobResult
from pipeline_runner import pipeline_runner

router = APIRouter()

INPUTS_DIR = Path("inputs")
INPUTS_DIR.mkdir(exist_ok=True)

@router.post("/", response_model=JobResponse)
async def create_job(
    file: UploadFile = File(...),
    direction: str = Form("LR"),
    camera_angle: str = Form("LR"),  # NEW: LR, RL, or TOP
    db: Session = Depends(get_db)
):
    # Validate camera_angle
    if camera_angle not in ["LR", "RL", "TOP"]:
        raise HTTPException(status_code=400, detail="Invalid camera_angle. Must be LR, RL, or TOP")
    
    # Save uploaded video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = INPUTS_DIR / filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Create DB entry with camera_angle
    db_job = Job(
        video_path=str(file_path),
        direction=direction,
        camera_angle=camera_angle,  # NEW
        status="created"
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    # Start processing in background
    pipeline_runner.start_job(db_job.id)
    
    return db_job

@router.get("/", response_model=List[JobResponse])
def read_jobs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    jobs = db.query(Job).order_by(Job.created_at.desc()).offset(skip).limit(limit).all()
    return jobs

@router.get("/latest", response_model=JobResponse)
def read_latest_job(db: Session = Depends(get_db)):
    """Get the most recently created job"""
    job = db.query(Job).order_by(Job.created_at.desc()).first()
    if job is None:
        raise HTTPException(status_code=404, detail="No jobs found")
    return job

@router.get("/{job_id}", response_model=JobResponse)
def read_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/{job_id}/result")
def read_job_result(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if not job.result_json_path or not Path(job.result_json_path).exists():
        if job.status == "completed":
             raise HTTPException(status_code=404, detail="Result file missing despite completion")
        raise HTTPException(status_code=400, detail="Job not completed yet")
        
    import json
    print(f"[DEBUG] Reading result from: {job.result_json_path}")
    with open(job.result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"[DEBUG] Loaded data keys: {list(data.keys())}")
    if "pipeline_info" in data:
        print(f"[DEBUG] Pipeline info: {data['pipeline_info']}")
        
    # Map raw data to JobResult schema if needed, or just return dict which Pydantic handles partially
    # For now, just sending back raw structure wrapped
    
    
    # We might need to transform the data structure for the frontend
    # But let's assume the frontend adapts to the pipeline logic for now
    
    # Actually, simpler is to just let frontend fetch the JSON file directly via the /outputs static mount.
    # But this endpoint is good for metadata.
    
    # Return the FULL json content directly at root level
    # The data already has: pipeline_type, summary, stage4_damage, wagons, etc.
    # Frontend expects these at root level, not wrapped in another 'summary' key
    return data