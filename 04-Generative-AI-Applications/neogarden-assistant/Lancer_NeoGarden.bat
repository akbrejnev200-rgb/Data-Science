@echo off
cd /d "%~dp0"
echo Lancement de l'assistant NeoGarden...

rem Choix de l'interpreteur Python (dans cet ordre) :
rem   1. la variable NEOGARDEN_PYTHON (chemin complet de python.exe), si definie ;
rem   2. le .venv du projet, s'il existe ;
rem   3. le "python" du PATH.
if defined NEOGARDEN_PYTHON (
    set "PY=%NEOGARDEN_PYTHON%"
) else if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

rem Rend le paquet rag_garden (dossier src) importable sans installation.
set "PYTHONPATH=%~dp0src"

"%PY%" -m streamlit run app\streamlit_app.py
echo.
echo L'application s'est arretee. Appuyez sur une touche pour fermer.
pause
