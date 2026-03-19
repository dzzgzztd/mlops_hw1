import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ClearMLService:
    def __init__(self) -> None:
        self.is_configured = self._is_env_configured()

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
        if not self.is_configured:
            logger.warning("ClearML is not configured")
            return None

        try:
            from clearml import Task
        except Exception as e:
            logger.error("Failed to import ClearML Task: %s", e)
            return None

        try:
            task = Task.create(
                project_name=os.getenv("CLEARML_PROJECT", "ML-Service"),
                task_name=f"{model_type}_training",
                task_type=Task.TaskTypes.training,
            )

            task.add_tags([model_type, "automated"])

            if dataset_name:
                task.set_parameter("General/dataset_name", dataset_name)

            for key, value in hyperparameters.items():
                task.set_parameter(f"Hyperparameters/{key}", value)

            output_uri = os.getenv("CLEARML_OUTPUT_URI")
            if output_uri:
                try:
                    task.output_uri = output_uri
                    logger.info("ClearML output_uri configured: %s", output_uri)
                except Exception as e:
                    logger.warning(
                        "Could not configure ClearML output_uri=%s: %s",
                        output_uri,
                        e,
                    )

            # ВАЖНО: переводим draft -> in_progress
            task.started(force=True)

            logger.info("ClearML experiment created: %s", task.id)
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
        if task is None:
            return

        try:
            task_logger = task.get_logger()
            for metric_name, metric_value in metrics.items():
                task_logger.report_scalar(
                    title="metrics",
                    series=metric_name,
                    value=float(metric_value),
                    iteration=iteration,
                )
        except Exception as e:
            logger.exception("Failed to log metrics to ClearML: %s", e)

    def register_model(
        self,
        task: Any,
        model_path: str,
        model_name: str,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        if task is None:
            return

        try:
            from clearml import OutputModel

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

    def finalize_task(self, task: Any, failed: bool = False) -> None:
        if task is None:
            return

        try:
            task.flush(wait_for_uploads=True)
        except Exception:
            logger.exception("Failed to flush ClearML task")

        try:
            if failed:
                task.mark_failed(status_reason="Training failed", force=True)
            else:
                task.mark_completed(force=True)
                task.close()
            logger.info("ClearML task finalized: %s", getattr(task, "id", None))
        except Exception as e:
            logger.exception("Failed to finalize ClearML task: %s", e)


clearml_service = ClearMLService()