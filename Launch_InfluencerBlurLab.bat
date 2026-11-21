@echo off
setlocal
title Influencer Blur Lab

cd /d "%~dp0"

echo.
echo ======================================================
echo   Influencer Blur Lab
echo ======================================================
echo.

rem 1. Auto-pull latest updates from GitHub (safe if offline)
where git >nul 2>&1
if errorlevel 1 goto :skip_git
if not exist ".git" goto :skip_git

echo [1/3] Checking GitHub for latest updates...
git pull --ff-only
if errorlevel 1 echo       Could not reach GitHub or local edits present - continuing with your local copy.
goto :find_python

:skip_git
echo [1/3] Skipping GitHub update check.

:find_python
set "PY_CMD="
py -3.11 --version >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.11"
if not defined PY_CMD (
    py -3 --version >nul 2>&1
    if not errorlevel 1 set "PY_CMD=py -3"
)
if not defined PY_CMD (
    python --version >nul 2>&1
    if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
    echo.
    echo ERROR: Python 3 was not found on this PC.
    echo Please install Python 3.11 from https://www.python.org/downloads/
    pause
    exit /b 1
)

rem 2. Create .venv if missing
if exist ".venv\Scripts\python.exe" goto :check_reqs

echo [2/3] Setting up clean local Python environment...
%PY_CMD% -m venv .venv
if errorlevel 1 (
    echo ERROR: Could not create .venv folder.
    pause
    exit /b 1
)

:check_reqs
".venv\Scripts\python.exe" -c "import hashlib, pathlib, sys; cur=hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest(); p=pathlib.Path('.venv/.req_hash'); sys.exit(0 if p.exists() and p.read_text().strip()==cur else 1)"
if not errorlevel 1 goto :reqs_ok

echo [2/3] Installing / updating required packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "import hashlib, pathlib; pathlib.Path('.venv/.req_hash').write_text(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest())"
goto :start_app

:reqs_ok
echo [2/3] Packages already up to date.

:start_app
echo [3/3] Starting Influencer Blur Lab in your web browser...
echo.
".venv\Scripts\python.exe" -m app.server --host 127.0.0.1 --port 8785 %*

if errorlevel 1 (
    echo.
    echo Influencer Blur Lab stopped unexpectedly.
    pause
)
endlocal
