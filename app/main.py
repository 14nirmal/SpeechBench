"""
SpeechBench FastAPI Application Entrypoint
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_transcribe import router as transcribe_router
from app.api.routes_evaluate import router as evaluate_router
from app.api.routes_benchmark import router as benchmark_router
from app.database.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure database tables are initialized on startup."""
    init_db()
    yield


app = FastAPI(
    title="SpeechBench ASR Intelligence API",
    description="ASR benchmarking, evaluation metrics (WER, CER, RTF, Entity Error Rate), and latency analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(transcribe_router)
app.include_router(evaluate_router)
app.include_router(benchmark_router)


@app.get("/")
def root():
    return {
        "service": "SpeechBench",
        "version": "1.0.0",
        "description": "Speech Recognition Evaluation and Benchmarking Platform",
        "docs": "/docs",
    }


@app.get("/health")
def healthcheck():
    return {"status": "healthy"}
