from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

class JobBase(BaseModel):
    video_path: str
    direction: str = "LR"

class JobCreate(JobBase):
    pass

class JobUpdate(BaseModel):
    status: Optional[str] = None
    progress: Optional[float] = None
    current_stage: Optional[str] = None
    total_wagons: Optional[int] = None
    defects_count: Optional[int] = None
    error_message: Optional[str] = None
    output_dir: Optional[str] = None

class JobResponse(JobBase):
    id: int
    status: str
    progress: float
    current_stage: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_dir: Optional[str] = None
    camera_angle: Optional[str] = "LR"
    total_wagons: int
    defects_count: int
    live_counts: Optional[Dict[str, Any]] = {}
    error_message: Optional[str] = None

    class Config:
        from_attributes = True

# For the dashboard/analysis pages, we might return the raw JSON from the pipeline
# or specific structures. For now, let's keep it flexible.
class JobResult(BaseModel):
    job_id: int
    summary: Dict[str, Any]
    wagons: List[Dict[str, Any]]