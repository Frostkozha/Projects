@echo off
rem Start the Brain service (Qwen3.5-9B Q4_K_M, one slot, 4,096 tokens) with readiness probes.
rem It refuses to start if port 8080 is busy - close the chat server (start-llama-menu.bat) first.
title Brain service (local Qwen3.5-9B)
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m brain.cli start --config config\brain-local-9b.yaml
pause
