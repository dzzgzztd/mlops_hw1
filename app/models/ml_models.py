import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Абстрактный базовый класс для всех ML моделей"""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model: Union[BaseEstimator, None] = None
        self.is_trained = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **hyperparameters) -> Dict[str, Any]:
        """Обучение модели с заданными гиперпараметрами"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание на новых данных"""
        pass

    def save(self, path: str) -> None:
        """Сохранение модели на диск"""
        if self.model is None:
            raise ValueError("Модель не обучена, невозможно сохранить")
        joblib.dump(self.model, path)
        logger.info(f"Модель сохранена в {path}")

    def load(self, path: str) -> None:
        """Загрузка модели с диска"""
        self.model = joblib.load(path)
        self.is_trained = True
        logger.info(f"Модель загружена из {path}")


class SklearnLogisticRegression(BaseModel):
    """Класс для логистической регрессии"""

    def __init__(self):
        super().__init__("logistic_regression")
        self.default_params = {
            'C': 1.0,
            'max_iter': 100,
            'random_state': 42
        }

    def fit(self, X: np.ndarray, y: np.ndarray, **hyperparameters) -> Dict[str, Any]:
        """Обучение логистической регрессии"""
        try:
            # Объединяем параметры по умолчанию с переданными
            params = {**self.default_params, **hyperparameters}

            self.model = LogisticRegression(**params)
            self.model.fit(X, y)
            self.is_trained = True

            # Вычисляем accuracy на обучающей выборке
            train_accuracy = self.model.score(X, y)

            logger.info(f"LogisticRegression обучена с параметрами: {params}")
            logger.info(f"Accuracy на обучающей выборке: {train_accuracy:.4f}")

            return {
                "status": "success",
                "model_type": self.model_type,
                "hyperparameters": params,
                "train_accuracy": train_accuracy
            }

        except Exception as e:
            logger.error(f"Ошибка при обучении LogisticRegression: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание"""
        if not self.is_trained or self.model is None:
            raise ValueError("Модель не обучена")
        return self.model.predict(X)


class SklearnRandomForest(BaseModel):
    """Класс для случайного леса"""

    def __init__(self):
        super().__init__("random_forest")
        self.default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42
        }

    def fit(self, X: np.ndarray, y: np.ndarray, **hyperparameters) -> Dict[str, Any]:
        """Обучение случайного леса"""
        try:
            # Объединяем параметры по умолчанию с переданными
            params = {**self.default_params, **hyperparameters}

            self.model = RandomForestClassifier(**params)
            self.model.fit(X, y)
            self.is_trained = True

            # Вычисляем accuracy на обучающей выборке
            train_accuracy = self.model.score(X, y)

            logger.info(f"RandomForest обучен с параметрами: {params}")
            logger.info(f"Accuracy на обучающей выборке: {train_accuracy:.4f}")

            return {
                "status": "success",
                "model_type": self.model_type,
                "hyperparameters": params,
                "train_accuracy": train_accuracy
            }

        except Exception as e:
            logger.error(f"Ошибка при обучении RandomForest: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание"""
        if not self.is_trained or self.model is None:
            raise ValueError("Модель не обучена")
        return self.model.predict(X)


# Фабрика для создания моделей
class ModelFactory:
    """Фабрика для создания экземпляров моделей"""

    _model_classes = {
        "logistic_regression": SklearnLogisticRegression,
        "random_forest": SklearnRandomForest
    }

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict]:
        """Возвращает список доступных моделей и их параметров"""
        models_info = {}
        for name, model_class in cls._model_classes.items():
            # Создаем временный экземпляр чтобы получить default_params
            instance = model_class()
            models_info[name] = {
                "default_hyperparameters": instance.default_params,
                "description": f"Scikit-learn {name.replace('_', ' ').title()}"
            }
        return models_info

    @classmethod
    def create_model(cls, model_type: str) -> BaseModel:
        """Создает экземпляр модели по типу"""
        if model_type not in cls._model_classes:
            raise ValueError(f"Неизвестный тип модели: {model_type}. Доступные: {list(cls._model_classes.keys())}")
        return cls._model_classes[model_type]()