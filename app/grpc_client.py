import grpc
import logging
from app.grpc.generated import ml_service_pb2, ml_service_pb2_grpc


def run():
    """Тестовый клиент для gRPC сервера"""

    # Настройка логгера
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Подключение к серверу
    channel = grpc.insecure_channel('localhost:50051')
    stub = ml_service_pb2_grpc.MLServiceStub(channel)

    try:
        print("Тестирование gRPC сервера...")

        # 1. Health Check
        print("\n1. Health Check:")
        health_response = stub.HealthCheck(ml_service_pb2.HealthRequest())
        print(f"   Status: {health_response.status}")
        print(f"   Message: {health_response.message}")

        # 2. Доступные модели
        print("\n2. Доступные модели:")
        models_response = stub.GetAvailableModels(ml_service_pb2.AvailableModelsRequest())
        if models_response.status == "success":
            for model_type, model_info in models_response.available_models.items():
                print(f"   - {model_type}: {model_info.description}")
        else:
            print(f"   Error: {models_response.error}")

        # 3. Обучение модели
        print("\n3. Обучение Logistic Regression:")
        train_response = stub.TrainModel(ml_service_pb2.TrainModelRequest(
            model_type="logistic_regression",
            hyperparameters={"C": "0.1", "max_iter": "200"}
        ))
        if train_response.status == "success":
            print(f"   Model ID: {train_response.model_id}")
            print(f"   Accuracy: {train_response.train_accuracy:.4f}")
            model_id = train_response.model_id
        else:
            print(f"   Error: {train_response.error}")
            return

        # 4. Список моделей
        print("\n4. Список обученных моделей:")
        list_response = stub.ListModels(ml_service_pb2.ListModelsRequest())
        if list_response.status == "success":
            for model in list_response.models:
                print(f"   - {model.model_id} ({model.model_type})")
        else:
            print(f"   Error: {list_response.error}")

        # 5. Предсказание
        print("\n5. Тестирование предсказания:")
        predict_response = stub.Predict(ml_service_pb2.PredictRequest(
            model_id=model_id,
            data=[
                ml_service_pb2.DataRow(features=[1.0, 2.0, 3.0, 4.0]),
                ml_service_pb2.DataRow(features=[5.0, 6.0, 7.0, 8.0])
            ]
        ))
        if predict_response.status == "success":
            print(f"   Predictions: {predict_response.predictions}")
        else:
            print(f"   Error: {predict_response.error}")

        # 6. Датасеты
        print("\n6. Список датасетов:")
        datasets_response = stub.ListDatasets(ml_service_pb2.ListDatasetsRequest())
        if datasets_response.status == "success":
            for dataset in datasets_response.datasets:
                print(f"   - {dataset}")
        else:
            print(f"   Error: {datasets_response.error}")

        print("\ngRPC тестирование завершено!")

    except grpc.RpcError as e:
        logger.error(f"gRPC error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    run()