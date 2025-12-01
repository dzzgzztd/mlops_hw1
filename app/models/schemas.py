from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class ModelTrainRequest(BaseModel):
    """Схема для запроса на обучение модели с датасетом"""
    model_type: str = Field(..., description="Тип модели (logistic_regression, random_forest)")
    dataset_name: Optional[str] = Field(None, description="Имя датасета из DVC (опционально)")
    hyperparameters: Optional[Dict[str, Any]] = Field(default={}, description="Гиперпараметры модели")

    class Config:
        schema_extra = {
            "example": {
                "model_type": "logistic_regression",
                "dataset_name": "iris",
                "hyperparameters": {"C": 0.1, "max_iter": 200}
            }
        }


class ModelRetrainRequest(BaseModel):
    """Схема для запроса на переобучение модели"""
    dataset_name: Optional[str] = Field(None, description="Имя датасета из DVC (опционально)")
    hyperparameters: Optional[Dict[str, Any]] = Field(default={}, description="Новые гиперпараметры")

    class Config:
        schema_extra = {
            "example": {
                "dataset_name": "iris",
                "hyperparameters": {"C": 0.5, "max_iter": 300}
            }
        }


# Обновленные схемы для ответов
class ModelTrainResponse(BaseModel):
    """Схема для ответа на обучение модели"""
    status: str
    model_id: Optional[str] = None
    model_type: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    train_accuracy: Optional[float] = None
    model_path: Optional[str] = None
    dataset_info: Optional[str] = None
    clearml_task_id: Optional[str] = None
    error: Optional[str] = None


class DatasetInfoResponse(BaseModel):
    """Схема для ответа с информацией о датасете"""
    status: str
    name: Optional[str] = None
    data: Optional[List[Dict]] = None
    columns: Optional[List[str]] = None
    shape: Optional[List[int]] = None
    description: Optional[str] = None
    error: Optional[str] = None


class ModelPredictRequest(BaseModel):
    """Схема для запроса на предсказание"""
    data: List[List[float]] = Field(..., description="Данные для предсказания в виде 2D массива")

    class Config:
        schema_extra = {
            "example": {
                "data": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
            }
        }


class ModelPredictResponse(BaseModel):
    """Схема для ответа на предсказание"""
    status: str
    model_id: Optional[str] = None
    model_type: Optional[str] = None
    predictions: Optional[List[int]] = None
    error: Optional[str] = None


class ModelListResponse(BaseModel):
    """Схема для ответа со списком моделей"""
    status: str
    models: List[Dict[str, Any]]
    count: int


class ModelInfoResponse(BaseModel):
    """Схема для ответа с информацией о модели"""
    status: str
    model_id: Optional[str] = None
    model_type: Optional[str] = None
    is_trained: Optional[bool] = None
    error: Optional[str] = None


class ModelDeleteResponse(BaseModel):
    """Схема для ответа на удаление модели"""
    status: str
    message: Optional[str] = None
    error: Optional[str] = None


class AvailableModelsResponse(BaseModel):
    """Схема для ответа со списком доступных моделей"""
    status: str
    available_models: Dict[str, Dict[str, Any]]


class HealthResponse(BaseModel):
    """Схема для health check"""
    status: str
    message: str
    timestamp: str


class DatasetListResponse(BaseModel):
    """Схема для ответа со списком датасетов"""
    status: str
    datasets: List[Dict[str, Any]]
    count: int
