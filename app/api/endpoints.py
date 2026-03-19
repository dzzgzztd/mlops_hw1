import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import (
    DatasetLoadError,
    DatasetNotFoundError,
    ModelDeleteError,
    ModelNotFoundError,
    PredictionError,
    TrainingError,
)
from app.api.schemas import (
    ModelTrainRequest,
    ModelTrainResponse,
    ModelPredictRequest,
    ModelPredictResponse,
    ModelRetrainRequest,
    ModelListResponse,
    ModelInfoResponse,
    ModelDeleteResponse,
    AvailableModelsResponse,
    HealthResponse,
    DatasetListResponse,
    DatasetInfoResponse,
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
    return HealthResponse(
        status="healthy",
        message="ML Service is running",
        timestamp=datetime.now().isoformat(),
    )


@models_router.get("/models/available", response_model=AvailableModelsResponse)
async def get_available_models():
    try:
        result = model_service.get_available_models()
        return AvailableModelsResponse(status="success", **result)
    except Exception as e:
        logger.exception("Error in get_available_models: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@models_router.post("/models/train", response_model=ModelTrainResponse)
async def train_model(request: ModelTrainRequest):
    try:
        result = model_service.train_model(
            model_type=request.model_type,
            dataset_name=request.dataset_name,
            hyperparameters=request.hyperparameters or {},
        )
        return ModelTrainResponse(status="success", **result)

    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except (DatasetLoadError, TrainingError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Error in train_model: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@models_router.get("/models", response_model=ModelListResponse)
async def list_trained_models():
    try:
        result = model_service.list_models()
        return ModelListResponse(status="success", **result)
    except Exception as e:
        logger.exception("Error in list_models: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@models_router.get("/models/{model_id}", response_model=ModelInfoResponse)
async def get_model_info(model_id: str):
    try:
        result = model_service.get_model_info(model_id)
        return ModelInfoResponse(status="success", **result)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Error in get_model_info: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@models_router.post("/models/{model_id}/predict", response_model=ModelPredictResponse)
async def predict(model_id: str, request: ModelPredictRequest):
    try:
        import numpy as np

        X = np.array(request.data)
        result = model_service.get_prediction(model_id, X)
        return ModelPredictResponse(status="success", **result)

    except ModelNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PredictionError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Error in predict: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@models_router.post("/models/{model_id}/retrain", response_model=ModelTrainResponse)
@models_router.put("/models/{model_id}/retrain", response_model=ModelTrainResponse)
async def retrain_model(model_id: str, request: ModelRetrainRequest):
    try:
        result = model_service.retrain_model(
            model_id=model_id,
            dataset_name=request.dataset_name,
            hyperparameters=request.hyperparameters or {},
        )
        return ModelTrainResponse(status="success", **result)

    except ModelNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except (DatasetLoadError, TrainingError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Error in retrain_model: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@models_router.delete("/models/{model_id}", response_model=ModelDeleteResponse)
async def delete_model(model_id: str):
    try:
        result = model_service.delete_model(model_id)
        return ModelDeleteResponse(status="success", **result)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ModelDeleteError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Error in delete_model: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@datasets_router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets():
    try:
        result = dataset_service.list_datasets()
        return DatasetListResponse(**result)
    except Exception as e:
        logger.exception("Error in list_datasets: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@datasets_router.get("/datasets/{dataset_name}", response_model=DatasetInfoResponse)
async def get_dataset(dataset_name: str):
    try:
        result = dataset_service.get_dataset(dataset_name)
        return DatasetInfoResponse(**result)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except DatasetLoadError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Error in get_dataset: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@datasets_router.post("/datasets/{dataset_name}/pull")
@datasets_router.post("/datasets/{dataset_name}/update")
async def pull_dataset(dataset_name: str):
    try:
        result = dataset_service.pull_dataset(dataset_name)
        return {"status": "success", **result}
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except DatasetLoadError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Error in pull_dataset: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
