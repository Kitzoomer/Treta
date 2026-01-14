@echo off
title TRETA - Panel de Control
cd /d C:\treta

REM Si existe venv, úsalo (recomendado)
if exist ".venv\Scripts\pythonw.exe" (
  ".venv\Scripts\pythonw.exe" "C:\treta\treta_app.py"
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "C:\treta\treta_app.py"
) else (
  REM Si no hay venv, usa el launcher de Python
  py "C:\treta\treta_app.py"
)
