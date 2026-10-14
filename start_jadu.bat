@echo off
rem Starts Jadu: Ollama (if needed), the server, and the app window.
setlocal
cd /d "%~dp0"
title Jadu

if not exist "venv\Scripts\python.exe" (
  echo Could not find venv\Scripts\python.exe.
  echo Create the virtual environment first - see "Setup" in README.md.
  pause
  exit /b 1
)

"venv\Scripts\python.exe" web.py
if errorlevel 1 (
  echo.
  echo Jadu stopped with an error. Read the message above.
  pause
)
