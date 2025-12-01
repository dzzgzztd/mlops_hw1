# MLOps-hw1
### Якушева Арина, БПИ228

---
**Выполнено:**
- **REST API** (FastAPI) - все основные операции с моделями
- **gRPC-операции**
- **Веб-дашборд** (Gradio)
- **Docker контейнеризация** 
- **2 типа ML моделей** - Logistic Regression и Random Forest (взяты из scipy)
- **Управление жизненным циклом** - обучение, предсказание, переобучение, удаление
- **DVC** - версионирование датасетов
- **MinIO** - хранилище артефактов
- **Автоматическое логирование** - все операции логируются

---
**Запуск:**
```
make build
make up
make init-dvc
make status
```
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