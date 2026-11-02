@echo off
echo Starting SpeechBench FastAPI REST Backend...
call .\venv\Scripts\activate
uvicorn app.main:app --port 8000 --reload
pause
