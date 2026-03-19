"""
Демонстрация ML Service Platform с DVC и ClearML
"""

import requests

API_BASE = "http://localhost:8000/api/v1"


def print_step(step, description):
    print(f"\n{'=' * 60}")
    print(f"ШАГ {step}: {description}")
    print(f"{'=' * 60}")


def demo_final():
    """Полная демонстрация работы системы с DVC и ClearML"""

    print("ДЕМОНСТРАЦИЯ ML SERVICE PLATFORM")

    # Шаг 1: Проверка здоровья
    print_step(1, "Проверка здоровья сервиса")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"Статус: {health['status']}")
            print(f"Сообщение: {health['message']}")
        else:
            print(f"Сервис недоступен: {response.status_code}")
            return
    except Exception as e:
        print(f"Ошибка подключения: {e}")
        return

    # Шаг 2: Доступные модели
    print_step(2, "Получение списка доступных моделей")
    response = requests.get(f"{API_BASE}/models/available")
    if response.status_code == 200:
        models = response.json()
        print("Доступные модели:")
        for model_type, info in models['available_models'].items():
            print(f"  - {model_type}: {info['description']}")
    else:
        print(f"Ошибка: {response.text}")
        return

    # Шаг 3: Список датасетов из DVC
    print_step(3, "Получение списка датасетов из DVC")
    response = requests.get(f"{API_BASE}/datasets")
    if response.status_code == 200:
        datasets = response.json()
        print("Доступные датасеты:")
        for dataset in datasets['datasets']:
            dvc_info = " (DVC)" if dataset.get('dvc_tracked') else ""
            print(
                f"  - {dataset['name']}{dvc_info}: {dataset.get('rows', 0)} строк, {dataset.get('columns', 0)} колонок")
    else:
        print(f"Ошибка: {response.text}")
        return

    # Шаг 4: Обучение модели на реальном датасете
    print_step(4, "Обучение Logistic Regression на датасете iris")
    train_data = {
        "model_type": "logistic_regression",
        "dataset_name": "iris",
        "hyperparameters": {
            "C": 0.1,
            "max_iter": 200,
            "random_state": 42
        }
    }

    response = requests.post(f"{API_BASE}/models/train", json=train_data)
    if response.status_code == 200:
        result = response.json()
        if result['status'] == 'success':
            lr_model_id = result['model_id']
            print(f"   Модель обучена с ClearML!")
            print(f"   ID: {lr_model_id}")
            print(f"   Точность: {result['train_accuracy']:.4f}")
            print(f"   Датасет: {result['dataset_info']}")
            if result.get('clearml_task_id'):
                print(f"   ClearML Task ID: {result['clearml_task_id']}")
        else:
            print(f"Ошибка обучения: {result.get('error', 'Unknown')}")
            return
    else:
        print(f"HTTP ошибка: {response.status_code}")
        return

    # Шаг 5: Обучение Random Forest на другом датасете
    print_step(5, "Обучение Random Forest на датасете wine")
    train_data = {
        "model_type": "random_forest",
        "dataset_name": "wine",
        "hyperparameters": {
            "n_estimators": 50,
            "max_depth": 5,
            "random_state": 42
        }
    }

    response = requests.post(f"{API_BASE}/models/train", json=train_data)
    if response.status_code == 200:
        result = response.json()
        if result['status'] == 'success':
            rf_model_id = result['model_id']
            print(f"   Модель обучена с ClearML!")
            print(f"   ID: {rf_model_id}")
            print(f"   Точность: {result['train_accuracy']:.4f}")
            if result.get('clearml_task_id'):
                print(f"   ClearML Task ID: {result['clearml_task_id']}")
        else:
            print(f"Ошибка обучения: {result.get('error', 'Unknown')}")
    else:
        print(f"HTTP ошибка: {response.status_code}")

    # Шаг 6: Список всех моделей
    print_step(6, "Список всех обученных моделей")
    response = requests.get(f"{API_BASE}/models")
    if response.status_code == 200:
        result = response.json()
        print(f"Всего моделей: {result['count']}")
        for model in result['models']:
            print(f"  - {model['model_id']} ({model['model_type']})")

    # Шаг 7: Предсказание
    print_step(7, "Тестирование предсказания")
    predict_data = {
        "data": [
            [5.1, 3.5, 1.4, 0.2],  # Пример из iris датасета
            [6.7, 3.0, 5.2, 2.3]
        ]
    }

    response = requests.post(f"{API_BASE}/models/{lr_model_id}/predict", json=predict_data)
    if response.status_code == 200:
        result = response.json()
        if result['status'] == 'success':
            print(f"   Предсказания получены:")
            print(f"   Модель: {result['model_type']}")
            print(f"   Предсказания: {result['predictions']}")
        else:
            print(f"Ошибка предсказания: {result.get('error', 'Unknown')}")

    # Шаг 8: Переобучение модели
    print_step(8, "Переобучение модели с новыми параметрами")
    retrain_data = {
        "dataset_name": "iris",
        "hyperparameters": {
            "C": 0.5,
            "max_iter": 300
        }
    }

    response = requests.put(f"{API_BASE}/models/{lr_model_id}/retrain", json=retrain_data)
    if response.status_code == 200:
        result = response.json()
        if result['status'] == 'success':
            print(f"   Модель переобучена с ClearML!")
            print(f"   Новая точность: {result['train_accuracy']:.4f}")
            if result.get('clearml_task_id'):
                print(f"   ClearML Task ID: {result['clearml_task_id']}")
        else:
            print(f"Ошибка переобучения: {result.get('error', 'Unknown')}")

    print(f"\nИТОГИ РЕАЛИЗАЦИИ:")
    print(f"  REST API с Swagger: http://localhost:8000/docs")
    print(f"  Веб-дашборд: http://localhost:7860")
    print(f"  gRPC сервис: localhost:50051")
    print(f"  DVC датасеты: {len(datasets['datasets'])} датасетов версионированы")
    print(f"  ClearML интеграция: эксперименты и модели трекаются")
    print(f"  MinIO хранилище: http://localhost:9001")
    print(f"  ClearML UI: http://localhost:8080")
    print(f"  Обучено моделей: 2 с метриками в ClearML")


if __name__ == "__main__":
    demo_final()