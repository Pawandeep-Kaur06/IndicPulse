@echo off
rem Start the IndicPulse FastAPI server

rem Activate virtual environment if present
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

uvicorn src.api_fastapi:app --host 0.0.0.0 --port 8000
