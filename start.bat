@echo off
REM ============================================================
REM Wagon Inspection System - Startup Script
REM ============================================================
REM This script starts all three services required for the system:
REM   1. Chandra OCR Server (Port 8001)
REM   2. Backend API Server (Port 8000)
REM   3. Frontend Dev Server (Port 5173)
REM ============================================================

echo.
echo ============================================================
echo          WAGON INSPECTION SYSTEM - STARTING...
echo ============================================================
echo.

REM Get the directory where this script is located
set PROJECT_DIR=%~dp0

echo Starting Chandra OCR Server...
start "Chandra OCR Server (Port 8001)" cmd /k "cd /d %PROJECT_DIR% && call chandra_ocr_venv\Scripts\activate && python chandra_ocr_server.py"

REM Wait a moment for OCR server to initialize
timeout /t 3 /nobreak > nul

echo Starting Backend API Server...
start "Backend API (Port 8000)" cmd /k "cd /d %PROJECT_DIR% && call wagon_inspection_venv\Scripts\activate && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait a moment for backend to initialize
timeout /t 3 /nobreak > nul

echo Starting Frontend Dev Server...
start "Frontend (Port 5173)" cmd /k "cd /d %PROJECT_DIR%frontend && npm run dev"

echo.
echo ============================================================
echo          ALL SERVICES STARTED!
echo ============================================================
echo.
echo   Chandra OCR Server: http://localhost:8001
echo   Backend API:        http://localhost:8000
echo   Frontend:           http://localhost:5173
echo.
echo   Close this window or press any key to exit.
echo   (The services will continue running in their own windows)
echo ============================================================
echo.

pause
