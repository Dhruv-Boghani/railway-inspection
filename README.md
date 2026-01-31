# 🚂 Railway Wagon Inspection System

An AI-powered railway wagon inspection system that processes video footage to automatically detect wagons, assess image quality, enhance blur/low-light frames, detect door conditions, identify damage, and extract wagon numbers via OCR.

---
## Links of the project

frontend : https://railway-inspection-lovat.vercel.app/upload

backend : https://dhruvboghani-wegon-inspaction-server.hf.space

chandra-ocr : https://dhruvboghani-railway-inspection-try.hf.space

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [System Architecture](#-system-architecture)
4. [Technology Stack](#-technology-stack)
5. [Project Structure](#-project-structure)
6. [AI Pipeline Stages](#-ai-pipeline-stages)
7. [Installation](#-installation)
8. [Running the System](#-running-the-system)
9. [API Reference](#-api-reference)
10. [Model Weights](#-model-weights)
11. [Wagon Number Format](#-wagon-number-format)
12. [Screenshots](#-screenshots)

---

## 🎯 Overview

This system was developed to automate railway wagon inspection at yards and depots. Instead of manual visual inspection, camera footage from **side** and **top** angles is processed through a multi-stage AI pipeline to:

- **Count wagons** passing through the frame
- **Detect door conditions** (Good/Damaged/Missing)
- **Identify damage** (Dents/Scratches with severity levels)
- **Extract wagon numbers** (11-digit OCR with validation)
- **Generate inspection reports** with detailed analytics

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Dual Camera Support** | Side camera (LR/RL direction) and Top camera pipelines |
| **Real-time Processing** | WebSocket-based live progress updates |
| **Image Enhancement** | MPRNet deblurring for motion blur correction |
| **Smart OCR** | CRAFT text detection + Qwen2-VL OCR with check digit validation |
| **Blur vs Deblur Compare** | Side-by-side comparison showing enhancement improvements |
| **PDF Report Generation** | Downloadable inspection reports with analytics |
| **Interactive Analysis** | Zoom, pan, and per-wagon detailed view |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                       │
│                          http://localhost:5173                       │
│  ┌──────────┐ ┌────────────┐ ┌─────────┐ ┌──────────┐ ┌───────────┐ │
│  │  Upload  │ │ Processing │ │ Results │ │ Analysis │ │  Compare  │ │
│  └────┬─────┘ └─────┬──────┘ └────┬────┘ └────┬─────┘ └─────┬─────┘ │
└───────│─────────────│─────────────│───────────│─────────────│───────┘
        │             │ WebSocket   │           │             │
        ▼             ▼             ▼           ▼             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI + SQLite)                      │
│                          http://localhost:8000                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │   /jobs/ - Create, List, Delete jobs + WebSocket progress     │   │
│  │   /tools/ - Compare pipeline, single frame tools              │   │
│  │   /files/ - Static file serving                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────────┐                 ┌─────────────────────────┐
│  SIDE CAMERA PIPELINE │                 │   TOP CAMERA PIPELINE    │
│      (6 Stages)       │                 │       (5 Stages)         │
│  ┌─────────────────┐  │                 │  ┌──────────────────┐    │
│  │ 1. Detection    │  │                 │  │ 1. Detection     │    │
│  │ 2. Quality      │  │                 │  │ 2. Quality       │    │
│  │ 3. Enhancement  │  │                 │  │ 3. Enhancement   │    │
│  │ 4. Doors        │  │                 │  │ 4. Damage        │    │
│  │ 5. Damage       │  │                 │  │ 5. Aggregation   │    │
│  │ 6. OCR          │  │                 │  └──────────────────┘    │
│  └────────┬────────┘  │                 └─────────────────────────┘
│           │           │
└───────────│───────────┘
            │ HTTP POST
            ▼
┌─────────────────────────┐
│     OCR SERVER          │
│  http://localhost:8001  │
│  ┌───────────────────┐  │
│  │ Qwen2-VL-2B Model │  │
│  │ 8-bit quantized   │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

---

## 🛠️ Technology Stack

### Frontend
| Technology | Purpose |
|------------|---------|
| React 18 | UI Framework |
| Vite | Build tool & dev server |
| Tailwind CSS | Styling with custom Indigo/Violet theme |
| Recharts | Charts and graphs |
| Lucide React | Icons |
| html2pdf.js | PDF report generation |
| Axios | HTTP client |

### Backend
| Technology | Purpose |
|------------|---------|
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| SQLAlchemy | ORM for SQLite |
| WebSocket | Real-time progress updates |

### AI/ML
| Technology | Purpose |
|------------|---------|
| PyTorch | Deep learning framework |
| Ultralytics YOLO | Object detection (v8, v12) |
| MPRNet | Motion deblurring |
| CRAFT | Text region detection |
| Qwen2-VL-2B | OCR with vision-language model |
| timm | EfficientNet door classifier |
| OpenCV | Image processing |

---

## 📁 Project Structure

```
Wagon_Inspection_System/
│
├── 📁 backend/                      # FastAPI Backend
│   ├── main.py                      # App entry point
│   ├── database.py                  # SQLite configuration
│   ├── schemas.py                   # Pydantic models
│   ├── pipeline_runner.py           # Async job executor
│   └── routers/
│       ├── jobs.py                  # Job CRUD + WebSocket
│       ├── tools.py                 # Compare pipeline
│       └── files.py                 # Static file serving
│
├── 📁 frontend/                     # React Frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── UploadPage.jsx       # Video upload
│   │   │   ├── ProcessingPage.jsx   # Live progress
│   │   │   ├── ResultsPage.jsx      # Dashboard
│   │   │   ├── AnalysisPage.jsx     # Detailed view
│   │   │   ├── ReportPage.jsx       # Reports list
│   │   │   └── ComparePage.jsx      # Blur vs Deblur
│   │   ├── components/
│   │   │   ├── Sidebar.jsx          # Navigation
│   │   │   └── Layout.jsx           # Page wrapper
│   │   └── context/
│   │       └── JobContext.jsx       # Global state
│   └── tailwind.config.js           # Theme colors
│
├── 📁 modules/                      # AI Pipeline Modules
│   ├── wagon_detection_counting.py  # Stage 1: SIDE detection
│   ├── top_wagon_detection.py       # Stage 1: TOP detection
│   ├── quality_assessment.py        # Stage 2: Quality check
│   ├── image_enhancement.py         # Stage 3: MPRNet deblur
│   ├── door_detection_classification.py  # Stage 4: Doors
│   ├── damage_detection.py          # Stage 5: Damage
│   ├── top_damage_detection.py      # Stage 4 TOP: Damage
│   ├── wagon_number_extraction.py   # Stage 6: OCR
│   ├── craft_text_detection.py      # CRAFT wrapper
│   └── chandra_ocr_bridge.py        # OCR HTTP client
│
├── 📁 models/weights/               # ML Model Weights (~1.4GB)
│   ├── MPRNET.pth                   # Deblur model
│   ├── craft_mlt_25k.pth            # Text detection
│   ├── yolo12s_wagon_detection.pt   # Wagon detection
│   ├── yolo12n_Door_ROI_best.pt     # Door detection
│   ├── few_shot_classifier.pth      # Door classifier
│   ├── yolo8s-seg_damage_best.pt    # Damage segmentation
│   ├── top_detection_best.pt        # TOP wagon detection
│   └── top_damage_best.pt           # TOP damage detection
│
├── 📄 main.py                       # SIDE pipeline orchestrator
├── 📄 main_top_pipeline.py          # TOP pipeline orchestrator
├── 📄 pipeline_config.py            # Configuration
├── 📄 chandra_ocr_server.py         # OCR FastAPI server
├── 📄 chandra_ocr_service.py        # OCR logic
│
├── 📄 requirements_main.txt         # Main env dependencies
├── 📄 requirements_ocr.txt          # OCR env dependencies
├── 📄 start.bat                     # Windows startup script
│
├── 📁 inputs/                       # Uploaded videos
├── 📁 outputs/                      # Processing outputs
└── 📄 wagon_inspection.db           # SQLite database
```

---

## 🤖 AI Pipeline Stages

### SIDE Camera Pipeline (6 Stages)

| Stage | Module | Model | Output |
|-------|--------|-------|--------|
| **1. Detection** | `wagon_detection_counting.py` | YOLOv12s + ByteTrack | Wagon crops, counts |
| **2. Quality** | `quality_assessment.py` | Laplacian/Tenengrad | Quality scores, best frames |
| **3. Enhancement** | `image_enhancement.py` | MPRNet | Deblurred frames |
| **4. Doors** | `door_detection_classification.py` | YOLOv12n + EfficientNet | Door status per wagon |
| **5. Damage** | `damage_detection.py` | YOLOv8-seg | Damage masks, severity |
| **6. OCR** | `wagon_number_extraction.py` | CRAFT + Qwen2-VL | 11-digit wagon numbers |

### TOP Camera Pipeline (5 Stages)

| Stage | Module | Model | Output |
|-------|--------|-------|--------|
| **1. Detection** | `top_wagon_detection.py` | YOLOv12-TOP | Wagon crops from above |
| **2. Quality** | `quality_assessment.py` | Laplacian/Tenengrad | Quality scores |
| **3. Enhancement** | `image_enhancement.py` | MPRNet | Enhanced frames |
| **4. Damage** | `top_damage_detection.py` | YOLOv8-seg-TOP | Roof damage |
| **5. Aggregation** | - | - | Final results.json |

---

## 💻 Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (recommended) or CPU
- ~5GB disk space for models and environments

### Step 1: Clone Repository
```bash
git clone <your-repo-url> Wagon_Inspection_System
cd Wagon_Inspection_System
```

### Step 2: Create Main Pipeline Environment
```powershell
# Create virtual environment
python -m venv wagon_inspection_venv
wagon_inspection_venv\Scripts\activate

# Install PyTorch with CUDA (or CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For CPU only: pip install torch torchvision

# Install dependencies
pip install -r requirements_main.txt
deactivate
```

### Step 3: Create OCR Environment
```powershell
python -m venv chandra_ocr_venv
chandra_ocr_venv\Scripts\activate

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements_ocr.txt
deactivate
```

### Step 4: Install Frontend
```powershell
cd frontend
npm install
cd ..
```

### Step 5: Download Model Weights
Place the following files in `models/weights/`:
- `MPRNET.pth` (~1GB)
- `craft_mlt_25k.pth` (~83MB)
- `yolo12s_wagon_detection_&_counting_best.pt` (~19MB)
- `yolo12n_Door_ROI_best.pt` (~5.5MB)
- `few_shot_classifier_door_best_model.pth` (~56MB)
- `yolo8s-seg_damage_best.pt` (~24MB)
- `top_detection_best.pt` (~19MB)
- `top_damage_best.pt` (~19MB)
- `RealESRGAN_x4plus.pth` (~67MB)

---

## 🚀 Running the System

### Quick Start (Windows)
```batch
# Just double-click:
start.bat
```

### Manual Start (3 Terminals)

**Terminal 1: Backend API**
```powershell
wagon_inspection_venv\Scripts\activate
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2: OCR Server**
```powershell
chandra_ocr_venv\Scripts\activate
python chandra_ocr_server.py
```

**Terminal 3: Frontend**
```powershell
cd frontend
npm run dev
```

### Access URLs
| Service | URL |
|---------|-----|
| **Frontend** | http://localhost:5173 |
| **Backend API** | http://localhost:8000 |
| **API Docs** | http://localhost:8000/docs |
| **OCR Server** | http://localhost:8001 |

---

## 🐳 Docker Deployment

### Prerequisites
- Docker Desktop installed
- Docker Compose v2.0+
- Model weights in `models/weights/` folder

### Quick Start with Docker
```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Docker Services
| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| **backend** | wagon-backend | 8000 | FastAPI + AI Pipeline |
| **ocr-server** | wagon-ocr | 8001 | Qwen2-VL OCR |
| **frontend** | wagon-frontend | 5173 | React UI |

### GPU Support (Optional)
To enable NVIDIA GPU acceleration:
1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Uncomment the GPU sections in `docker-compose.yml`
3. Rebuild: `docker-compose up --build`

### Volume Mounts
| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./models` | `/app/models` | ML model weights |
| `./inputs` | `/app/inputs` | Uploaded videos |
| `./outputs` | `/app/outputs` | Processing results |

---

## 📡 API Reference

### Jobs Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/jobs/` | Create new inspection job |
| `GET` | `/jobs/` | List all jobs |
| `GET` | `/jobs/{id}` | Get job details |
| `GET` | `/jobs/{id}/result` | Get results JSON |
| `DELETE` | `/jobs/{id}` | Delete job |
| `WS` | `/jobs/{id}/ws` | WebSocket for live progress |

### Tools Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/tools/compare-pipeline` | Blur vs Deblur comparison |

---

## 📦 Model Weights

| Model | File | Size | Purpose |
|-------|------|------|---------|
| MPRNet | `MPRNET.pth` | ~1GB | Motion deblurring |
| CRAFT | `craft_mlt_25k.pth` | ~83MB | Text region detection |
| YOLOv12s | `yolo12s_wagon_*.pt` | ~19MB | Wagon detection |
| YOLOv12n | `yolo12n_Door_*.pt` | ~5.5MB | Door detection |
| EfficientNet | `few_shot_classifier_*.pth` | ~56MB | Door classification |
| YOLOv8-seg | `yolo8s-seg_damage_*.pt` | ~24MB | Damage segmentation |
| Qwen2-VL | HuggingFace (8-bit) | ~4GB | OCR recognition |

---

## 🔢 Wagon Number Format

Indian Railway 11-digit wagon numbering system:

```
Position:  C1  C2  C3  C4  C5  C6  C7  C8  C9  C10 C11
Example:   3   1   0   3   0   3   6   0   5   0   5
           └────┘   └────┘   └────┘   └────────┘   │
           Type     Railway   Year     Serial     Check
```

| Positions | Field | Example | Meaning |
|-----------|-------|---------|---------|
| C1-C2 | Wagon Type | 31 | Covered Wagon |
| C3-C4 | Railway Code | 03 | Northern Railway |
| C5-C6 | Year | 03 | 2003 |
| C7-C10 | Serial | 6050 | Individual number |
| C11 | Check Digit | 5 | Validation |

### Wagon Type Codes
| Code Range | Type |
|------------|------|
| 10-29 | Open Wagon |
| 30-39 | Covered Wagon |
| 40-54 | Tank Wagon |
| 55-69 | Flat Wagon |
| 70-79 | Hopper Wagon |
| 80-84 | Well Wagon |
| 85-89 | Brake Van |

### Railway Codes
| Code | Railway |
|------|---------|
| 01 | Central Railway |
| 02 | Eastern Railway |
| 03 | Northern Railway |
| 04 | North East Railway |
| 05 | Northeast Frontier Railway |
| 06 | Southern Railway |
| 07 | South Eastern Railway |
| 08 | Western Railway |
| ... | ... |

---

## 📸 Frontend Pages

| Page | Description |
|------|-------------|
| **Upload** | Drag-drop video, select camera type (Side/Top) and direction (LR/RL) |
| **Processing** | Real-time progress with stage indicators and wagon count |
| **Results** | Charts showing door status, damage summary, OCR success rate |
| **Analysis** | Per-wagon detailed view with zoom, pan, and annotated images |
| **Compare** | Upload single blur frame to compare Blur vs Deblur detection |
| **Reports** | Job history with PDF download option |

---

## 🎨 Theme Colors

The UI uses an Indigo/Violet color scheme:

| Color | Hex | Usage |
|-------|-----|-------|
| Primary | `#2B2E6D` | Headers, buttons |
| Primary Light | `#4C4FB3` | Hover states |
| Primary Dark | `#1E1F4B` | Sidebar |
| Accent | `#7C6CF2` | Highlights |
| Success | `#22C55E` | Good doors, valid OCR |
| Warning | `#F59E0B` | Damaged doors |
| Danger | `#EF4444` | Missing doors, errors |

---

## 👥 Team

**Developed by TEAM HACKHUSTLERS**

© 2026 All Rights Reserved

---

## 📄 License

This project is proprietary software developed for railway wagon inspection automation.
