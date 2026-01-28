from fastapi import APIRouter, HTTPException
from pathlib import Path
import os

router = APIRouter()

# Primarily used for browsing outputs if needed
# The StaticFiles mount in main.py handles actual serving.

@router.get("/browse/{job_id}")
def browse_job_files(job_id: int):
    # This could list all available images for a job to help frontend gallery
    pass