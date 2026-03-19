# MLOps-hw1
### Якушева Арина, БПИ228

---
**Выполнено:**
- REST API для управления моделями
- gRPC-сервис и gRPC-клиент
- интерактивный dashboard
- 2 типа моделей:
  - Logistic Regression
  - Random Forest
- обучение, предсказание, переобучение, удаление моделей
- список доступных моделей
- health-check endpoint
- список датасетов и их обновление из DVC
- DVC + MinIO для хранения и версионирования датасетов
- ClearML для логирования экспериментов и моделей
- запуск в Docker Compose
- запуск в Minikube / Kubernetes

---
**Запуск:**
```
make build
make up
make init-dvc
```
---
**Kubernetes:**
```
make k8s-up
```
API: http://127.0.0.1:8000/docs

Dashboard: http://127.0.0.1:7860

---
**Документация:**

После запуска: http://localhost:8000/docs

Основные эндпоинты:
- GET /api/v1/health - статус сервиса
- GET /api/v1/models/available - доступные типы моделей
- POST /api/v1/models/train - обучение новой модели
- GET /api/v1/models - список обученных моделей
- POST /api/v1/models/{id}/predict - предсказание
- PUT /api/v1/models/{id}/retrain - переобучение
- DELETE /api/v1/models/{id} - удаление модели
- GET /api/v1/datasets - список датасетов с DVC информацией
- GET /api/v1/datasets/{name} - информация о датасете
- POST /api/v1/datasets/{name}/pull - обновить из DVC

---
**Проверка gRPC-сервера**

Порт: 50051

Есть файлик grpc_client.py, в нем базовые тесты