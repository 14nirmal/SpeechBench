@echo off
echo Starting SpeechBench Interactive Evaluation Dashboard...
call .\venv\Scripts\activate
streamlit run app\dashboard\app.py --server.port 8501
pause
