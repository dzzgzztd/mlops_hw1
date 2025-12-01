from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.logger import setup_logger
from app.api.endpoints import health_router, models_router, datasets_router

logger = setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Dict[str, Any], None]:
    """Lifespan для управления событиями жизненного цикла"""
    logger.info("ML Service API starting up...")

    startup_data = {"message": "API started successfully"}

    yield startup_data

    logger.info("ML Service API shutting down...")


app = FastAPI(
    title="ML Service API",
    description="REST API для обучения и управления ML моделями",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(datasets_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "ML Service API",
        "version": "1.0.0",
        "docs": "/docs",
        "health_check": "/api/v1/"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
