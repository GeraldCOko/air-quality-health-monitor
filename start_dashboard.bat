@echo off
rem Double-click this file to open the dashboard. Close this window to stop it.
cd /d "%~dp0"
echo Starting the Air Quality and Health Monitor ...
echo Your browser will open at http://localhost:8501 in a few seconds.
echo Keep this window open while you use the dashboard. Close it to stop.
start "" /min cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py
pause
