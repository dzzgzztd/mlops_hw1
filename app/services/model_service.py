import logging
import uuid
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.core.exceptions import (
    DatasetLoadError,
    ModelDeleteError,
    ModelNotFoundError,
    PredictionError,
    TrainingError,
)
from app.models.ml_models import BaseModel, ModelFactory
from app.services.clearml_service import clearml_service
from app.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)


class ModelService:
    """Сервис для управления моделями"""

    def __init__(self, models_dir: str = "saved_models"):
        self.models: Dict[str, BaseModel] = {}
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.dataset_service = DatasetService()
        logger.info("ModelService инициализирован")

    def get_available_models(self) -> Dict[str, Any]:
        models_info = ModelFactory.get_available_models()
        logger.info("Запрошен список доступных моделей")
        return {
            "available_models": models_info,
        }

    def _load_training_data(self, dataset_name: str = None):
        if dataset_name:
            dataset_result = self.dataset_service.get_dataset(dataset_name)
            data = dataset_result["data"]

            if not data or len(data) == 0:
                raise DatasetLoadError(f"Датасет {dataset_name} пустой")

            df = pd.DataFrame(data)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            dataset_info = f"Датасет: {dataset_name} ({len(X)} samples)"
            return X, y, dataset_info

        from app.utils.data_generator import get_sample_data

        X, y = get_sample_data()
        dataset_info = f"Демо данные ({len(X)} samples)"
        return X, y, dataset_info

    def train_model(
        self,
        model_type: str,
        dataset_name: str = None,
        hyperparameters: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        clearml_task = None

        try:
            if hyperparameters is None:
                hyperparameters = {}

            clearml_task = clearml_service.create_experiment(
                model_type=model_type,
                hyperparameters=hyperparameters,
                dataset_name=dataset_name,
            )

            X, y, dataset_info = self._load_training_data(dataset_name)

            model = ModelFactory.create_model(model_type)
            model_id = str(uuid.uuid4())[:8]

            train_result = model.fit(X, y, **hyperparameters)

            if train_result["status"] != "success":
                raise TrainingError(
                    train_result.get("error", "Ошибка обучения модели")
                )

            model_path = self.models_dir / f"{model_id}_{model_type}.joblib"
            model.save(str(model_path))

            self.models[model_id] = model

            metrics = {
                "train_accuracy": float(train_result.get("train_accuracy", 0.0)),
                "dataset_rows": float(X.shape[0]),
                "dataset_features": float(X.shape[1]),
            }

            if clearml_task is not None:
                clearml_service.log_metrics(clearml_task, metrics)
                clearml_service.register_model(
                    clearml_task,
                    str(model_path),
                    f"{model_type}_{model_id}",
                    metrics,
                )
                clearml_service.finalize_task(clearml_task, failed=False)

            logger.info("Модель %s (%s) успешно обучена", model_id, model_type)

            return {
                "model_id": model_id,
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "train_accuracy": train_result.get("train_accuracy"),
                "model_path": str(model_path),
                "dataset_info": dataset_info,
                "clearml_task_id": getattr(clearml_task, "id", None) if clearml_task else None,
            }

        except Exception:
            if clearml_task is not None:
                try:
                    clearml_service.finalize_task(clearml_task, failed=True)
                except Exception:
                    logger.exception("Не удалось закрыть ClearML task после ошибки")
            raise

    def get_prediction(self, model_id: str, X: np.ndarray) -> Dict[str, Any]:
        if model_id not in self.models:
            raise ModelNotFoundError(f"Модель с ID {model_id} не найдена")

        model = self.models[model_id]

        try:
            predictions = model.predict(X)
        except Exception as e:
            raise PredictionError(
                f"Ошибка при получении предсказания от модели {model_id}: {e}"
            ) from e

        logger.info("Получены предсказания от модели %s", model_id)

        return {
            "model_id": model_id,
            "model_type": model.model_type,
            "predictions": predictions.tolist(),
        }

    def retrain_model(
        self,
        model_id: str,
        dataset_name: str = None,
        hyperparameters: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        clearml_task = None

        if model_id not in self.models:
            raise ModelNotFoundError(f"Модель с ID {model_id} не найдена")

        model = self.models[model_id]

        try:
            if hyperparameters is None:
                hyperparameters = {}

            clearml_task = clearml_service.create_experiment(
                model_type=model.model_type,
                hyperparameters=hyperparameters,
                dataset_name=dataset_name,
            )

            X, y, _ = self._load_training_data(dataset_name)

            train_result = model.fit(X, y, **hyperparameters)

            if train_result["status"] != "success":
                raise TrainingError(
                    train_result.get("error", "Ошибка переобучения модели")
                )

            model_path = self.models_dir / f"{model_id}_{model.model_type}.joblib"
            model.save(str(model_path))

            metrics = {
                "train_accuracy": float(train_result.get("train_accuracy", 0.0)),
                "dataset_rows": float(X.shape[0]),
                "dataset_features": float(X.shape[1]),
            }

            if clearml_task is not None:
                clearml_service.log_metrics(clearml_task, metrics)
                clearml_service.register_model(
                    clearml_task,
                    str(model_path),
                    f"{model.model_type}_{model_id}_retrained",
                    metrics,
                )
                clearml_service.finalize_task(clearml_task, failed=False)

            logger.info("Модель %s успешно переобучена", model_id)

            return {
                "model_id": model_id,
                "model_type": model.model_type,
                "hyperparameters": hyperparameters,
                "train_accuracy": train_result.get("train_accuracy"),
                "clearml_task_id": getattr(clearml_task, "id", None) if clearml_task else None,
            }

        except Exception:
            if clearml_task is not None:
                try:
                    clearml_service.finalize_task(clearml_task, failed=True)
                except Exception:
                    logger.exception("Не удалось закрыть ClearML task после ошибки")
            raise

    def delete_model(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self.models:
            raise ModelNotFoundError(f"Модель с ID {model_id} не найдена")

        model = self.models.pop(model_id)

        model_path = self.models_dir / f"{model_id}_{model.model_type}.joblib"
        try:
            if model_path.exists():
                model_path.unlink()
        except Exception as e:
            raise ModelDeleteError(
                f"Ошибка при удалении файла модели {model_id}: {e}"
            ) from e

        logger.info("Модель %s удалена", model_id)

        return {
            "message": f"Модель {model_id} успешно удалена",
        }

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self.models:
            raise ModelNotFoundError(f"Модель с ID {model_id} не найдена")

        model = self.models[model_id]

        return {
            "model_id": model_id,
            "model_type": model.model_type,
            "is_trained": model.is_trained,
        }

    def list_models(self) -> Dict[str, Any]:
        models_list = []
        for model_id, model in self.models.items():
            models_list.append(
                {
                    "model_id": model_id,
                    "model_type": model.model_type,
                    "is_trained": model.is_trained,
                }
            )

        logger.info("Запрошен список моделей. Найдено %s моделей", len(models_list))

        return {
            "models": models_list,
            "count": len(models_list),
        }