@echo off
rem Start the full local tutor (gate -> retriever -> Brain -> verifier -> chat page) on http://127.0.0.1:8000
rem It starts its own Brain on port 8080: close the llama chat launcher first.
title Histology Study Tutor (local development)
cd /d "%~dp0.."
start "" cmd /c "timeout /t 45 >nul & start http://127.0.0.1:8000/"
".venv\Scripts\python.exe" -m tutor_app.cli start --config config\local-dev.yaml
pause
