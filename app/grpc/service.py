import os
import sys
from concurrent import futures
from datetime import datetime
import grpc
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.exceptions import (
    DatasetLoadError,
    DatasetNotFoundError,
    ModelDeleteError,
    ModelNotFoundError,
    PredictionError,
    TrainingError,
)
from app.grpc.generated import ml_service_pb2, ml_service_pb2_grpc
from app.services.dataset_service import DatasetService
from app.services.model_service import ModelService
from app.utils.logger import setup_logger

logger = setup_logger("ml_service_grpc", "INFO")


class MLServiceServicer(ml_service_pb2_grpc.MLServiceServicer):
    """gRPC сервис для работы с ML моделями"""

    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "saved_models")

        self.model_service = ModelService(models_dir=models_dir)
        self.dataset_service = DatasetService()
        logger.info("gRPC MLServiceServicer инициализирован")

    def HealthCheck(self, request, context):
        logger.info("gRPC HealthCheck called")
        return ml_service_pb2.HealthResponse(
            status="healthy",
            message="gRPC ML Service is running",
            timestamp=datetime.now().isoformat(),
        )

    def GetAvailableModels(self, request, context):
        try:
            logger.info("gRPC GetAvailableModels called")
            result = self.model_service.get_available_models()

            available_models = {}
            for model_type, model_info in result["available_models"].items():
                str_hyperparams = {
                    k: str(v)
                    for k, v in model_info["default_hyperparameters"].items()
                }
                hyperparams = ml_service_pb2.ModelHyperparameters(
                    parameters=str_hyperparams
                )
                available_models[model_type] = ml_service_pb2.ModelInfo(
                    model_type=model_type,
                    default_hyperparameters=hyperparams,
                    description=model_info["description"],
                )

            return ml_service_pb2.AvailableModelsResponse(
                status="success",
                available_models=available_models,
            )

        except Exception as e:
            logger.exception("Error in gRPC GetAvailableModels: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.AvailableModelsResponse(
                status="error",
                error=str(e),
            )

    def TrainModel(self, request, context):
        try:
            logger.info("gRPC TrainModel called: %s", request.model_type)

            hyperparameters = {
                k: self._convert_hyperparameter_value(v)
                for k, v in request.hyperparameters.items()
            }

            dataset_name = request.dataset_name if request.dataset_name else None

            result = self.model_service.train_model(
                model_type=request.model_type,
                dataset_name=dataset_name,
                hyperparameters=hyperparameters,
            )

            str_hyperparams = {
                k: str(v) for k, v in result["hyperparameters"].items()
            }

            logger.info("Model trained successfully: %s", result["model_id"])

            return ml_service_pb2.TrainModelResponse(
                status="success",
                model_id=result["model_id"],
                model_type=result["model_type"],
                hyperparameters=str_hyperparams,
                train_accuracy=result.get("train_accuracy", 0.0),
                model_path=result.get("model_path", ""),
            )

        except DatasetNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_service_pb2.TrainModelResponse(status="error", error=str(e))

        except (DatasetLoadError, TrainingError, ValueError) as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return ml_service_pb2.TrainModelResponse(status="error", error=str(e))

        except Exception as e:
            logger.exception("Error in gRPC TrainModel: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.TrainModelResponse(status="error", error=str(e))

    def ListModels(self, request, context):
        try:
            logger.info("gRPC ListModels called")
            result = self.model_service.list_models()

            models = []
            for model in result["models"]:
                models.append(
                    ml_service_pb2.ModelSummary(
                        model_id=model["model_id"],
                        model_type=model["model_type"],
                        is_trained=model["is_trained"],
                    )
                )

            return ml_service_pb2.ListModelsResponse(
                status="success",
                models=models,
                count=result["count"],
            )

        except Exception as e:
            logger.exception("Error in gRPC ListModels: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.ListModelsResponse(
                status="error",
                error=str(e),
            )

    def GetModelInfo(self, request, context):
        try:
            logger.info("gRPC GetModelInfo called: %s", request.model_id)
            result = self.model_service.get_model_info(request.model_id)

            return ml_service_pb2.ModelInfoResponse(
                status="success",
                model_id=result["model_id"],
                model_type=result["model_type"],
                is_trained=result["is_trained"],
            )

        except ModelNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_service_pb2.ModelInfoResponse(status="error", error=str(e))

        except Exception as e:
            logger.exception("Error in gRPC GetModelInfo: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.ModelInfoResponse(status="error", error=str(e))

    def Predict(self, request, context):
        try:
            logger.info("gRPC Predict called: %s", request.model_id)

            X = np.array([[feature for feature in row.features] for row in request.data])
            result = self.model_service.get_prediction(request.model_id, X)

            return ml_service_pb2.PredictResponse(
                status="success",
                model_id=result["model_id"],
                model_type=result["model_type"],
                predictions=result["predictions"],
            )

        except ModelNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_service_pb2.PredictResponse(status="error", error=str(e))

        except PredictionError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return ml_service_pb2.PredictResponse(status="error", error=str(e))

        except Exception as e:
            logger.exception("Error in gRPC Predict: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.PredictResponse(status="error", error=str(e))

    def RetrainModel(self, request, context):
        try:
            logger.info("gRPC RetrainModel called: %s", request.model_id)

            hyperparameters = {
                k: self._convert_hyperparameter_value(v)
                for k, v in request.hyperparameters.items()
            }

            dataset_name = request.dataset_name if request.dataset_name else None

            result = self.model_service.retrain_model(
                model_id=request.model_id,
                dataset_name=dataset_name,
                hyperparameters=hyperparameters,
            )

            str_hyperparams = {
                k: str(v) for k, v in result["hyperparameters"].items()
            }

            return ml_service_pb2.RetrainModelResponse(
                status="success",
                model_id=result["model_id"],
                model_type=result["model_type"],
                hyperparameters=str_hyperparams,
                train_accuracy=result.get("train_accuracy", 0.0),
            )

        except ModelNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_service_pb2.RetrainModelResponse(status="error", error=str(e))

        except DatasetNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_service_pb2.RetrainModelResponse(status="error", error=str(e))

        except (DatasetLoadError, TrainingError, ValueError) as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return ml_service_pb2.RetrainModelResponse(status="error", error=str(e))

        except Exception as e:
            logger.exception("Error in gRPC RetrainModel: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.RetrainModelResponse(status="error", error=str(e))

    def DeleteModel(self, request, context):
        try:
            logger.info("gRPC DeleteModel called: %s", request.model_id)
            result = self.model_service.delete_model(request.model_id)

            return ml_service_pb2.DeleteModelResponse(
                status="success",
                message=result["message"],
            )

        except ModelNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_service_pb2.DeleteModelResponse(status="error", error=str(e))

        except ModelDeleteError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return ml_service_pb2.DeleteModelResponse(status="error", error=str(e))

        except Exception as e:
            logger.exception("Error in gRPC DeleteModel: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.DeleteModelResponse(status="error", error=str(e))

    def ListDatasets(self, request, context):
        try:
            logger.info("gRPC ListDatasets called")

            result = self.dataset_service.list_datasets()
            datasets = [dataset["name"] for dataset in result["datasets"]]

            return ml_service_pb2.ListDatasetsResponse(
                status="success",
                datasets=datasets,
                count=result["count"],
            )

        except Exception as e:
            logger.exception("Error in gRPC ListDatasets: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.ListDatasetsResponse(
                status="error",
                error=str(e),
            )

    @staticmethod
    def _convert_hyperparameter_value(value: str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(
        MLServiceServicer(), server
    )

    port = "50051"
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("gRPC Server started on port %s", port)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("gRPC Server stopped")
        server.stop(0)


if __name__ == "__main__":
    serve()