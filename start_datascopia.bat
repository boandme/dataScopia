@echo off
setlocal

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.10+ and try again.
    pause
    exit /b 1
)

python -c "import flask" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies from requirements.txt ...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed. See the errors above.
        pause
        exit /b 1
    )
)

python run.py
if errorlevel 1 (
    echo.
    echo DataScopia exited with an error. See the output above.
    pause
)
