# ============================================================
# CS6422 Sports Prediction App
# First-Time Environment Setup
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " CS6422 Environment Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/4] Checking Python..." -ForegroundColor Yellow

try {
    $pythonVersion = python --version
    Write-Host "Found $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: Python is not installed or not available in PATH." -ForegroundColor Red
    exit 1
}

# Check that we're in the project root
Write-Host ""
Write-Host "[2/4] Checking project files..." -ForegroundColor Yellow

if (-not (Test-Path "backend")) {
    Write-Host "ERROR: 'backend' directory not found." -ForegroundColor Red
    Write-Host "Make sure you run this script from the project root." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "backend\requirements.txt")) {
    Write-Host "ERROR: backend\requirements.txt not found." -ForegroundColor Red
    exit 1
}

Write-Host "Project structure found." -ForegroundColor Green

# Create virtual environment
Write-Host ""
Write-Host "[3/4] Creating virtual environment..." -ForegroundColor Yellow

if (Test-Path ".venv") {
    Write-Host ".venv already exists. Skipping creation." -ForegroundColor Green
}
else {
    python -m venv .venv

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }

    Write-Host "Virtual environment created." -ForegroundColor Green
}

# Install dependencies
Write-Host ""
Write-Host "[4/4] Installing dependencies..." -ForegroundColor Yellow

$python = ".\.venv\Scripts\python.exe"

& $python -m pip install --upgrade pip

& $python -m pip install -r backend\requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " SETUP COMPLETE" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Activate the virtual environment with:" -ForegroundColor Yellow
Write-Host ""
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""

Write-Host "Then start FastAPI with:" -ForegroundColor Yellow
Write-Host ""
Write-Host "    cd backend; python -m uvicorn app.main:app --reload" -ForegroundColor White
Write-Host ""

Write-Host "    http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host ""