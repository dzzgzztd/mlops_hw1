import grpc
from concurrent import futures
import numpy as np
from datetime import datetime
import os
import sys

# Добавляем путь к корню проекта для корректных импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.grpc.generated import ml_service_pb2, ml_service_pb2_grpc
from app.services.model_service import ModelService
from app.core.data_generator import SAMPLE_X, SAMPLE_Y

from app.core.logger import setup_logger

logger = setup_logger("ml_service_grpc", "INFO")


class MLServiceServicer(ml_service_pb2_grpc.MLServiceServicer):
    """gRPC сервис для работы с ML моделями"""

    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "saved_models")

        self.model_service = ModelService(models_dir=models_dir)
        logger.info("gRPC MLServiceServicer инициализирован")

    def HealthCheck(self, request, context):
        """Health check эндпоинт"""
        logger.info("gRPC HealthCheck called")
        return ml_service_pb2.HealthResponse(
            status="healthy",
            message="gRPC ML Service is running",
            timestamp=datetime.now().isoformat()
        )

    def GetAvailableModels(self, request, context):
        """Получение списка доступных моделей"""
        try:
            logger.info("gRPC GetAvailableModels called")
            result = self.model_service.get_available_models()

            if result["status"] == "error":
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(result["error"])
                return ml_service_pb2.AvailableModelsResponse(
                    status="error",
                    error=result["error"]
                )

            # Конвертируем результат в gRPC формат
            available_models = {}
            for model_type, model_info in result["available_models"].items():
                # Конвертируем значения гиперпараметров в строки для gRPC
                str_hyperparams = {k: str(v) for k, v in model_info["default_hyperparameters"].items()}
                hyperparams = ml_service_pb2.ModelHyperparameters(
                    parameters=str_hyperparams
                )
                available_models[model_type] = ml_service_pb2.ModelInfo(
                    model_type=model_type,
                    default_hyperparameters=hyperparams,
                    description=model_info["description"]
                )

            return ml_service_pb2.AvailableModelsResponse(
                status="success",
                available_models=available_models
            )

        except Exception as e:
            logger.error(f"Error in gRPC GetAvailableModels: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.AvailableModelsResponse(
                status="error",
                error=str(e)
            )

    def TrainModel(self, request, context):
        """Обучение модели"""
        try:
            logger.info(f"gRPC TrainModel called: {request.model_type}")

            # Конвертируем гиперпараметры
            hyperparameters = {
                k: self._convert_hyperparameter_value(v)
                for k, v in request.hyperparameters.items()
            }

            # Обучаем модель
            result = self.model_service.train_model(
                model_type=request.model_type,
                X=SAMPLE_X,
                y=SAMPLE_Y,
                hyperparameters=hyperparameters
            )

            if result["status"] == "error":
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(result["error"])
                return ml_service_pb2.TrainModelResponse(
                    status="error",
                    error=result["error"]
                )

            # Конвертируем гиперпараметры обратно в строки для gRPC
            str_hyperparams = {
                k: str(v) for k, v in result["hyperparameters"].items()
            }

            logger.info(f"Model trained successfully: {result['model_id']}")

            return ml_service_pb2.TrainModelResponse(
                status="success",
                model_id=result["model_id"],
                model_type=result["model_type"],
                hyperparameters=str_hyperparams,
                train_accuracy=result.get("train_accuracy", 0.0),
                model_path=result.get("model_path", "")
            )

        except Exception as e:
            logger.error(f"Error in gRPC TrainModel: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.TrainModelResponse(
                status="error",
                error=str(e)
            )

    def ListModels(self, request, context):
        """Список обученных моделей"""
        try:
            logger.info("gRPC ListModels called")
            result = self.model_service.list_models()

            if result["status"] == "error":
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(result["error"])
                return ml_service_pb2.ListModelsResponse(
                    status="error",
                    error=result["error"]
                )

            # Конвертируем модели
            models = []
            for model in result["models"]:
                models.append(ml_service_pb2.ModelSummary(
                    model_id=model["model_id"],
                    model_type=model["model_type"],
                    is_trained=model["is_trained"]
                ))

            logger.info(f"Returning {len(models)} models")

            return ml_service_pb2.ListModelsResponse(
                status="success",
                models=models,
                count=result["count"]
            )

        except Exception as e:
            logger.error(f"Error in gRPC ListModels: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.ListModelsResponse(
                status="error",
                error=str(e)
            )

    def GetModelInfo(self, request, context):
        """Информация о модели"""
        try:
            logger.info(f"gRPC GetModelInfo called: {request.model_id}")
            result = self.model_service.get_model_info(request.model_id)

            if result["status"] == "error":
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(result["error"])
                return ml_service_pb2.ModelInfoResponse(
                    status="error",
                    error=result["error"]
                )

            return ml_service_pb2.ModelInfoResponse(
                status="success",
                model_id=result["model_id"],
                model_type=result["model_type"],
                is_trained=result["is_trained"]
            )

        except Exception as e:
            logger.error(f"Error in gRPC GetModelInfo: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.ModelInfoResponse(
                status="error",
                error=str(e)
            )

    def Predict(self, request, context):
        """Предсказание"""
        try:
            logger.info(f"gRPC Predict called: {request.model_id}")

            X = np.array([[feature for feature in row.features] for row in request.data])

            result = self.model_service.get_prediction(request.model_id, X)

            if result["status"] == "error":
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(result["error"])
                return ml_service_pb2.PredictResponse(
                    status="error",
                    error=result["error"]
                )

            logger.info(f"Prediction successful for model: {request.model_id}")

            return ml_service_pb2.PredictResponse(
                status="success",
                model_id=result["model_id"],
                model_type=result["model_type"],
                predictions=result["predictions"]
            )

        except Exception as e:
            logger.error(f"Error in gRPC Predict: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.PredictResponse(
                status="error",
                error=str(e)
            )

    def RetrainModel(self, request, context):
        """Переобучение модели"""
        try:
            logger.info(f"gRPC RetrainModel called: {request.model_id}")

            # Конвертируем гиперпараметры
            hyperparameters = {
                k: self._convert_hyperparameter_value(v)
                for k, v in request.hyperparameters.items()
            }

            result = self.model_service.retrain_model(
                model_id=request.model_id,
                X=SAMPLE_X,
                y=SAMPLE_Y,
                hyperparameters=hyperparameters
            )

            if result["status"] == "error":
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(result["error"])
                return ml_service_pb2.RetrainModelResponse(
                    status="error",
                    error=result["error"]
                )

            # Конвертируем гиперпараметры обратно в строки для gRPC
            str_hyperparams = {
                k: str(v) for k, v in result["hyperparameters"].items()
            }

            logger.info(f"Model retrained successfully: {request.model_id}")

            return ml_service_pb2.RetrainModelResponse(
                status="success",
                model_id=result["model_id"],
                model_type=result["model_type"],
                hyperparameters=str_hyperparams,
                train_accuracy=result.get("train_accuracy", 0.0)
            )

        except Exception as e:
            logger.error(f"Error in gRPC RetrainModel: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.RetrainModelResponse(
                status="error",
                error=str(e)
            )

    def DeleteModel(self, request, context):
        """Удаление модели"""
        try:
            logger.info(f"gRPC DeleteModel called: {request.model_id}")
            result = self.model_service.delete_model(request.model_id)

            if result["status"] == "error":
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(result["error"])
                return ml_service_pb2.DeleteModelResponse(
                    status="error",
                    error=result["error"]
                )

            logger.info(f"Model deleted successfully: {request.model_id}")

            return ml_service_pb2.DeleteModelResponse(
                status="success",
                message=result["message"]
            )

        except Exception as e:
            logger.error(f"Error in gRPC DeleteModel: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.DeleteModelResponse(
                status="error",
                error=str(e)
            )

    def ListDatasets(self, request, context):
        """Список датасетов"""
        try:
            logger.info("gRPC ListDatasets called")

            datasets = ["iris", "wine", "breast_cancer", "digits"]

            return ml_service_pb2.ListDatasetsResponse(
                status="success",
                datasets=datasets,
                count=len(datasets)
            )

        except Exception as e:
            logger.error(f"Error in gRPC ListDatasets: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.ListDatasetsResponse(
                status="error",
                error=str(e)
            )

    def _convert_hyperparameter_value(self, value: str):
        """Конвертирует строковые значения гиперпараметров в правильные типы"""
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                # Оставляем как строку, если не удалось конвертировать
                return value


def serve():
    """Запуск gRPC сервера"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)

    # Порт для gRPC
    port = "50051"
    server.add_insecure_port(f"[::]:{port}")

    server.start()
    logger.info(f"gRPC Server started on port {port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("gRPC Server stopped")
        server.stop(0)


if __name__ == "__main__":
    serve()
