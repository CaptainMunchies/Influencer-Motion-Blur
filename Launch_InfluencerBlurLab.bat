@echo off
setlocal EnableDelayedExpansion
title Influencer Blur Lab

cd /d "%~dp0"

echo.
echo ======================================================
echo   Influencer Blur Lab
echo ======================================================
echo.

:: 1. Auto-pull latest updates from GitHub (safe if offline)
where git >nul 2>&1
if %errorlevel%==0 (
    if exist ".git" (
        echo [1/3] Checking GitHub for latest updates...
        git pull --ff-only
        if errorlevel 1 (
            echo       (Could not reach GitHub or local edits present — continuing with your local copy.)
        )
    )
) else (
    echo [1/3] Git not found — skipping auto-update check.
)

:: 2. Find Python 3.11 / Python 3 (avoiding Windows Store stub)
set "PY_CMD="
py -3.11 --version >nul 2>&1
if %errorlevel%==0 (
    set "PY_CMD=py -3.11"
) else (
    py -3 --version >nul 2>&1
    if %errorlevel%==0 (
        set "PY_CMD=py -3"
    ) else (
        python --version >nul 2>&1
        if %errorlevel%==0 (
            set "PY_CMD=python"
        )
    )
)

if "%PY_CMD%"=="" (
    echo.
    echo ERROR: Python 3 was not found on this PC.
    echo Please install Python 3.11 from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 3. Create .venv if missing
if not exist ".venv\Scripts\python.exe" (
    echo [2/3] Setting up clean local Python environment (.venv)...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create .venv folder.
        pause
        exit /b 1
    )
)

:: 4. Install or update requirements only when requirements.txt changed
set "REQ_HASH="
for /f "delims=" %%H in ('".venv\Scripts\python.exe" -c "import hashlib; print(hashlib.sha256(open('requirements.txt','rb').read()).hexdigest())"') do (
    set "REQ_HASH=%%H"
)

set "SAVED_HASH="
if exist ".venv\.req_hash" (
    set /p SAVED_HASH=<".venv\.req_hash"
)

if not "%REQ_HASH%"=="%SAVED_HASH%" (
    echo [2/3] Installing / updating required packages...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
    >".venv\.req_hash" echo %REQ_HASH%
) else (
    echo [2/3] Packages already up to date.
)

:: 5. Start the web app and auto-open in default browser
echo [3/3] Starting Influencer Blur Lab in your web browser...
echo.
".venv\Scripts\python.exe" -m app.server --host 127.0.0.1 --port 8765

if errorlevel 1 (
    echo.
    echo Influencer Blur Lab stopped unexpectedly.
    pause
)
endlocal
