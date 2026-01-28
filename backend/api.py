from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import os
import sys
from pathlib import Path

# Add parent directory to sys.path to allow importing from root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from database import engine, init_db
from routers import jobs, files, tools

# Create required directories
Path("outputs").mkdir(exist_ok=True)

# Initialize Database
init_db()

app = FastAPI(title="Wagon Inspection System API")

# CORS Setup
origins = [
    "http://localhost:5173",  # Vite default port
    "http://localhost:5174",  # Alternative Vite port
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory to serve images/videos
# This allows the frontend to access generated images directly
# e.g. http://localhost:8000/outputs/stage1/...
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Include Routers
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(files.router, prefix="/files", tags=["files"])
app.include_router(tools.router, prefix="/tools", tags=["tools"])

@app.get("/")
def read_root():
    return {"message": "Wagon Inspection System API is running"}