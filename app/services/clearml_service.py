import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ClearMLService:
    """Сервис для интеграции с ClearML"""

    def __init__(self) -> None:
        self.is_configured = self._is_env_configured()
        if self.is_configured:
            logger.info("ClearML configured via environment variables")
        else:
            logger.warning("ClearML env vars are missing, integration disabled")

    @staticmethod
    def _is_env_configured() -> bool:
        required = (
            "CLEARML_API_HOST",
            "CLEARML_WEB_HOST",
            "CLEARML_FILES_HOST",
            "CLEARML_API_ACCESS_KEY",
            "CLEARML_API_SECRET_KEY",
        )
        return all(os.getenv(key) for key in required)

    def create_experiment(
        self,
        model_type: str,
        hyperparameters: Dict[str, Any],
        dataset_name: Optional[str] = None,
    ) -> Optional[Any]:
        """Создание эксперимента в ClearML"""
        if not self.is_configured:
            logger.warning("ClearML is not configured")
            return None

        try:
            from clearml import Task
        except Exception as e:
            logger.error("Failed to import ClearML Task: %s", e)
            return None

        try:
            output_uri = os.getenv("CLEARML_OUTPUT_URI")

            task = Task.init(
                project_name=os.getenv("CLEARML_PROJECT", "ML-Service"),
                task_name=f"{model_type}_training",
                tags=[model_type, "automated"],
                reuse_last_task_id=False,
                output_uri=output_uri,
            )

            task.connect(hyperparameters, name="hyperparameters")

            if dataset_name:
                task.set_parameter("dataset_name", dataset_name)

            logger.info("ClearML experiment created: %s", getattr(task, "id", None))
            return task

        except Exception as e:
            logger.exception("Failed to create ClearML experiment: %s", e)
            return None

    def log_metrics(
        self,
        task: Any,
        metrics: Dict[str, float],
        iteration: int = 0,
    ) -> None:
        """Логирование метрик в ClearML"""
        if not self.is_configured or task is None:
            return

        try:
            task_logger = task.get_logger()
            for metric_name, metric_value in metrics.items():
                try:
                    value = float(metric_value)
                except (TypeError, ValueError):
                    logger.warning(
                        "Metric %s=%r cannot be converted to float, skipped",
                        metric_name,
                        metric_value,
                    )
                    continue

                task_logger.report_scalar(
                    title="metrics",
                    series=metric_name,
                    value=value,
                    iteration=iteration,
                )

            logger.info("Metrics logged to ClearML")
        except Exception as e:
            logger.exception("Failed to log metrics to ClearML: %s", e)

    def register_model(
        self,
        task: Any,
        model_path: str,
        model_name: str,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Регистрация модели в ClearML"""
        if not self.is_configured or task is None:
            return

        try:
            from clearml import OutputModel
        except Exception as e:
            logger.error("Failed to import ClearML OutputModel: %s", e)
            return

        try:
            output_model = OutputModel(
                task=task,
                name=model_name,
                framework="Scikit-Learn",
                tags=["automated", "ml-service"],
            )

            if metrics:
                output_model.update_design(
                    config_dict={"metrics": {k: float(v) for k, v in metrics.items()}}
                )

            output_model.update_weights(weights_filename=model_path)

            logger.info("Model registered in ClearML: %s", model_name)
        except Exception as e:
            logger.exception("Failed to register model in ClearML: %s", e)

    def finalize_task(self, task: Any, status: str = "completed") -> None:
        if not self.is_configured or task is None:
            return

        try:
            if status == "failed":
                try:
                    task.mark_failed(status_reason="Training failed")
                except Exception:
                    pass
            else:
                try:
                    task.mark_completed()
                except Exception:
                    pass

            task.close()
            logger.info("ClearML task finalized: %s", getattr(task, "id", None))
        except Exception as e:
            logger.exception("Failed to finalize ClearML task: %s", e)


clearml_service = ClearMLService()