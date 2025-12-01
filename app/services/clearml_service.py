import logging
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class ClearMLService:
    """Сервис для интеграции с ClearML"""

    def __init__(self):
        self.is_configured = False
        self._configure_clearml()

    def _configure_clearml(self):
        """Настройка ClearML конфигурации"""
        try:
            os.environ['CLEARML_API_HOST'] = 'http://localhost:8009'
            os.environ['CLEARML_WEB_HOST'] = 'http://localhost:8080'
            os.environ['CLEARML_FILES_HOST'] = 'http://localhost:8081'

            os.environ['CLEARML_API_ACCESS_KEY'] = 'minioadmin'
            os.environ['CLEARML_API_SECRET_KEY'] = 'minioadmin'

            self.is_configured = True
            logger.info("ClearML service configured")

        except Exception as e:
            logger.error(f"Failed to configure ClearML: {e}")
            self.is_configured = False

    def create_experiment(self, model_type: str, hyperparameters: Dict[str, Any],
                          dataset_name: str = None) -> Optional[Any]:
        """Создание эксперимента в ClearML"""
        if not self.is_configured:
            logger.warning("ClearML is not configured")
            return None

        try:
            from clearml import Task

            task = Task.init(
                project_name="ML-Service",
                task_name=f"{model_type}_training",
                tags=[model_type, "automated"]
            )

            # Логируем гиперпараметры
            task.connect(hyperparameters, name="hyperparameters")

            # Логируем информацию о датасете
            if dataset_name:
                task.set_parameter("dataset", dataset_name)

            logger.info(f"ClearML experiment created: {task.id}")
            return task

        except ImportError:
            logger.error("ClearML is not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to create ClearML experiment: {e}")
            return None

    def log_metrics(self, task, metrics: Dict[str, float]):
        """Логирование метрик"""
        if not self.is_configured or task is None:
            return

        try:
            for metric_name, metric_value in metrics.items():
                task.get_logger().report_scalar(
                    title="Metrics",
                    series=metric_name,
                    value=metric_value,
                    iteration=0
                )
            logger.debug("Metrics logged to ClearML")

        except Exception as e:
            logger.error(f"Failed to log metrics to ClearML: {e}")

    def register_model(self, task, model_path: str, model_name: str,
                       metrics: Dict[str, float] = None):
        """Регистрация модели в ClearML"""
        if not self.is_configured or task is None:
            return

        try:
            from clearml import OutputModel

            output_model = OutputModel(
                task=task,
                name=model_name,
                framework="Scikit-Learn",
                tags=["automated", "ml-service"]
            )

            # Добавляем метрики к модели
            if metrics:
                for metric_name, metric_value in metrics.items():
                    output_model.update_design(config_dict={f"metrics/{metric_name}": metric_value})

            # Загружаем файл модели
            output_model.update_weights(weights_filename=model_path)

            logger.info(f"Model registered in ClearML: {model_name}")

        except Exception as e:
            logger.error(f"Failed to register model in ClearML: {e}")


clearml_service = ClearMLService()