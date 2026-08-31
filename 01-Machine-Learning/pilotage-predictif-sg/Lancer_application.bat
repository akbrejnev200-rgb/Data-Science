@echo off
setlocal
cd /d "%~dp0"

set VENV_PYTHON=%~dp0..\venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo Environnement virtuel introuvable a l'emplacement : %VENV_PYTHON%
    pause
    exit /b 1
)

echo Demarrage de l'application...
start "" http://localhost:8050
"%VENV_PYTHON%" "%~dp0app\app.py"

pause
