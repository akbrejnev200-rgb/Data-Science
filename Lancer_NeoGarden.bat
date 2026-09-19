@echo off
cd /d "%~dp0"
echo Lancement de l'assistant NeoGarden...
"C:\Users\akbre\venvs\rag-jardin\Scripts\python.exe" -m streamlit run app.py
echo.
echo L'application s'est arretee. Appuyez sur une touche pour fermer.
pause
