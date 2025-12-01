import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, UploadFile, File

from app.models.schemas import (
    ModelTrainRequest, ModelTrainResponse,
    ModelPredictRequest, ModelPredictResponse,
    ModelRetrainRequest, ModelListResponse,
    ModelInfoResponse, ModelDeleteResponse,
    AvailableModelsResponse, HealthResponse,
    DatasetListResponse, DatasetInfoResponse
)
from app.services.model_service import ModelService
from app.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)

health_router = APIRouter(tags=["Health Check"])
models_router = APIRouter(tags=["Models"])
datasets_router = APIRouter(tags=["Datasets"])

model_service = ModelService()
dataset_service = DatasetService()


@health_router.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка статуса сервиса"""
    logger.info("Health check requested")
    return HealthResponse(
        status="healthy",
        message="ML Service is running",
        timestamp=datetime.now().isoformat()
    )


@models_router.get("/models/available", response_model=AvailableModelsResponse)
async def get_available_models():
    """Получить список доступных классов моделей"""
    try:
        logger.info("Request for available models")
        result = model_service.get_available_models()

        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        return AvailableModelsResponse(**result)

    except Exception as e:
        logger.error(f"Error in get_available_models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@models_router.post("/models/train", response_model=ModelTrainResponse)
async def train_model(request: ModelTrainRequest):
    """Обучить новую модель с выбором датасета"""
    try:
        logger.info(
            f"Training model: {request.model_type} with dataset: {request.dataset_name} and params: {request.hyperparameters}")

        result = model_service.train_model(
            model_type=request.model_type,
            dataset_name=request.dataset_name,
            hyperparameters=request.hyperparameters or {}
        )

        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

        return ModelTrainResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@models_router.get("/models", response_model=ModelListResponse)
async def list_trained_models():
    """Получить список всех обученных моделей"""
    try:
        logger.info("Request for model list")
        result = model_service.list_models()

        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        return ModelListResponse(**result)

    except Exception as e:
        logger.error(f"Error in list_models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@models_router.get("/models/{model_id}", response_model=ModelInfoResponse)
async def get_model_info(model_id: str):
    """Получить информацию о конкретной модели"""
    try:
        logger.info(f"Request for model info: {model_id}")
        result = model_service.get_model_info(model_id)

        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )

        return ModelInfoResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_model_info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@models_router.post("/models/{model_id}/predict", response_model=ModelPredictResponse)
async def predict_model(model_id: str, request: ModelPredictRequest):
    """Получить предсказание от модели"""
    try:
        logger.info(f"Prediction request for model: {model_id}")

        import numpy as np
        X = np.array(request.data)

        result = model_service.get_prediction(model_id, X)

        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

        return ModelPredictResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@models_router.put("/models/{model_id}/retrain", response_model=ModelTrainResponse)
async def retrain_model(model_id: str, request: ModelRetrainRequest):
    """Переобучить существующую модель с выбором датасета"""
    try:
        logger.info(
            f"Retraining model: {model_id} with dataset: {request.dataset_name} and params: {request.hyperparameters}")

        result = model_service.retrain_model(
            model_id=model_id,
            dataset_name=request.dataset_name,
            hyperparameters=request.hyperparameters or {}
        )

        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

        return ModelTrainResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in retrain_model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@models_router.delete("/models/{model_id}", response_model=ModelDeleteResponse)
async def delete_model(model_id: str):
    """Удалить модель"""
    try:
        logger.info(f"Delete request for model: {model_id}")
        result = model_service.delete_model(model_id)

        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )

        return ModelDeleteResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@datasets_router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets():
    """Получить список доступных датасетов"""
    try:
        logger.info("Request for dataset list")
        result = dataset_service.list_datasets()

        if result["status"] == "success":
            return DatasetListResponse(
                status="success",
                datasets=result["datasets"],
                count=result["count"]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

    except Exception as e:
        logger.error(f"Error in list_datasets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@datasets_router.get("/datasets/{dataset_name}")
async def get_dataset(dataset_name: str):
    """Получить информацию о конкретном датасете"""
    try:
        logger.info(f"Request for dataset: {dataset_name}")
        result = dataset_service.get_dataset(dataset_name)

        if result["status"] == "success":
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )

    except Exception as e:
        logger.error(f"Error in get_dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@datasets_router.post("/datasets/{dataset_name}/pull")
async def pull_dataset(dataset_name: str):
    """Обновить датасет из DVC"""
    try:
        logger.info(f"Pulling dataset from DVC: {dataset_name}")
        result = dataset_service.pull_dataset(dataset_name)

        if result["status"] == "success":
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

    except Exception as e:
        logger.error(f"Error in pull_dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@datasets_router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Загрузить новый датасет и добавить в DVC"""
    try:
        logger.info(f"Uploading dataset: {file.filename}")

        # Сохраняем временный файл
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Добавляем в DVC через сервис
        result = dataset_service.add_dataset(temp_path)

        # Удаляем временный файл
        os.unlink(temp_path)

        if result["status"] == "success":
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

    except Exception as e:
        logger.error(f"Error in upload_dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )