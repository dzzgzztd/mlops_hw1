import logging
import uuid
from typing import Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd

from app.models.ml_models import BaseModel, ModelFactory
from app.services.clearml_service import clearml_service
from app.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)


class ModelService:
    """Сервис для управления моделями"""

    def __init__(self, models_dir: str = "saved_models"):
        self.models: Dict[str, BaseModel] = {}  # model_id -> model_instance
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.dataset_service = DatasetService()
        logger.info("ModelService инициализирован")

    def get_available_models(self) -> Dict[str, Any]:
        """Возвращает список доступных классов моделей"""
        try:
            models_info = ModelFactory.get_available_models()
            logger.info("Запрошен список доступных моделей")
            return {
                "status": "success",
                "available_models": models_info
            }
        except Exception as e:
            logger.error(f"Ошибка при получении списка моделей: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def train_model(self, model_type: str, dataset_name: str = None,
                    hyperparameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Обучение новой модели"""
        try:
            if hyperparameters is None:
                hyperparameters = {}

            # Создаем эксперимент в ClearML
            clearml_task = clearml_service.create_experiment(
                model_type=model_type,
                hyperparameters=hyperparameters,
                dataset_name=dataset_name
            )

            # Загружаем данные из датасета или используем демо данные
            if dataset_name:
                dataset_result = self.dataset_service.get_dataset(dataset_name)
                if dataset_result['status'] == 'success':
                    data = dataset_result['data']
                    if data and len(data) > 0:
                        df = pd.DataFrame(data)
                        # Предполагаем, что последняя колонка - target
                        X = df.iloc[:, :-1].values
                        y = df.iloc[:, -1].values
                        dataset_info = f"Датасет: {dataset_name} ({len(X)} samples)"
                    else:
                        raise ValueError(f"Датасет {dataset_name} пустой")
                else:
                    raise ValueError(f"Ошибка загрузки датасета: {dataset_result.get('error')}")
            else:
                # Используем демо данные
                from app.core.data_generator import SAMPLE_X, SAMPLE_Y
                X, y = SAMPLE_X, SAMPLE_Y
                dataset_info = "Демо данные (100 samples)"

            # Создаем модель
            model = ModelFactory.create_model(model_type)
            model_id = str(uuid.uuid4())[:8]  # Короткий ID для удобства

            # Обучаем модель
            train_result = model.fit(X, y, **hyperparameters)

            if train_result["status"] == "success":
                # Сохраняем модель
                model_path = self.models_dir / f"{model_id}_{model_type}.joblib"
                model.save(str(model_path))

                # Сохраняем в памяти
                self.models[model_id] = model

                # Логируем метрики в ClearML
                metrics = {
                    "train_accuracy": train_result.get("train_accuracy", 0),
                    "dataset_rows": X.shape[0],
                    "dataset_features": X.shape[1]
                }
                clearml_service.log_metrics(clearml_task, metrics)

                # Регистрируем модель в ClearML
                clearml_service.register_model(
                    clearml_task,
                    str(model_path),
                    f"{model_type}_{model_id}",
                    metrics
                )

                logger.info(f"Модель {model_id} ({model_type}) успешно обучена")

                return {
                    "status": "success",
                    "model_id": model_id,
                    "model_type": model_type,
                    "hyperparameters": hyperparameters,
                    "train_accuracy": train_result.get("train_accuracy"),
                    "model_path": str(model_path),
                    "dataset_info": dataset_info,
                    "clearml_task_id": getattr(clearml_task, 'id', None) if clearml_task else None
                }
            else:
                return train_result

        except Exception as e:
            logger.error(f"Ошибка при обучении модели {model_type}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def get_prediction(self, model_id: str, X: np.ndarray) -> Dict[str, Any]:
        """Получение предсказания от модели"""
        try:
            if model_id not in self.models:
                return {
                    "status": "error",
                    "error": f"Модель с ID {model_id} не найдена"
                }

            model = self.models[model_id]
            predictions = model.predict(X)

            logger.info(f"Получены предсказания от модели {model_id}")

            return {
                "status": "success",
                "model_id": model_id,
                "model_type": model.model_type,
                "predictions": predictions.tolist()
            }

        except Exception as e:
            logger.error(f"Ошибка при получении предсказания от модели {model_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def retrain_model(self, model_id: str, dataset_name: str = None,
                      hyperparameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Переобучение существующей модели"""
        try:
            if model_id not in self.models:
                return {
                    "status": "error",
                    "error": f"Модель с ID {model_id} не найдена"
                }

            model = self.models[model_id]

            if hyperparameters is None:
                hyperparameters = {}

            # Создаем эксперимент в ClearML для переобучения
            clearml_task = clearml_service.create_experiment(
                model_type=model.model_type,
                hyperparameters=hyperparameters,
                dataset_name=dataset_name
            )

            # Загружаем данные
            if dataset_name:
                dataset_result = self.dataset_service.get_dataset(dataset_name)
                if dataset_result['status'] == 'success':
                    data = dataset_result['data']
                    df = pd.DataFrame(data)
                    X = df.iloc[:, :-1].values
                    y = df.iloc[:, -1].values
                else:
                    # Используем демо данные если датасет не найден
                    from app.core.data_generator import SAMPLE_X, SAMPLE_Y
                    X, y = SAMPLE_X, SAMPLE_Y
            else:
                # Используем демо данные
                from app.core.data_generator import SAMPLE_X, SAMPLE_Y
                X, y = SAMPLE_X, SAMPLE_Y

            # Переобучаем модель
            train_result = model.fit(X, y, **hyperparameters)

            if train_result["status"] == "success":
                # Сохраняем обновленную модель
                model_path = self.models_dir / f"{model_id}_{model.model_type}.joblib"
                model.save(str(model_path))

                # Логируем метрики в ClearML
                metrics = {
                    "train_accuracy": train_result.get("train_accuracy", 0),
                    "dataset_rows": X.shape[0],
                    "dataset_features": X.shape[1]
                }
                clearml_service.log_metrics(clearml_task, metrics)

                # Регистрируем обновленную модель в ClearML
                clearml_service.register_model(
                    clearml_task,
                    str(model_path),
                    f"{model.model_type}_{model_id}_retrained",
                    metrics
                )

                logger.info(f"Модель {model_id} успешно переобучена")

                return {
                    "status": "success",
                    "model_id": model_id,
                    "model_type": model.model_type,
                    "hyperparameters": hyperparameters,
                    "train_accuracy": train_result.get("train_accuracy"),
                    "clearml_task_id": getattr(clearml_task, 'id', None) if clearml_task else None
                }
            else:
                return train_result

        except Exception as e:
            logger.error(f"Ошибка при переобучении модели {model_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Удаление модели"""
        try:
            if model_id not in self.models:
                return {
                    "status": "error",
                    "error": f"Модель с ID {model_id} не найдена"
                }

            # Удаляем модель из памяти
            model = self.models.pop(model_id)

            # Удаляем файл модели
            model_path = self.models_dir / f"{model_id}_{model.model_type}.joblib"
            if model_path.exists():
                model_path.unlink()

            logger.info(f"Модель {model_id} удалена")

            return {
                "status": "success",
                "message": f"Модель {model_id} успешно удалена"
            }

        except Exception as e:
            logger.error(f"Ошибка при удалении модели {model_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Получение информации о модели"""
        try:
            if model_id not in self.models:
                return {
                    "status": "error",
                    "error": f"Модель с ID {model_id} не найдена"
                }

            model = self.models[model_id]

            return {
                "status": "success",
                "model_id": model_id,
                "model_type": model.model_type,
                "is_trained": model.is_trained
            }

        except Exception as e:
            logger.error(f"Ошибка при получении информации о модели {model_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def list_models(self) -> Dict[str, Any]:
        """Список всех обученных моделей"""
        try:
            models_list = []
            for model_id, model in self.models.items():
                models_list.append({
                    "model_id": model_id,
                    "model_type": model.model_type,
                    "is_trained": model.is_trained
                })

            logger.info(f"Запрошен список моделей. Найдено {len(models_list)} моделей")

            return {
                "status": "success",
                "models": models_list,
                "count": len(models_list)
            }

        except Exception as e:
            logger.error(f"Ошибка при получении списка моделей: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }