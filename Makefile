.PHONY: help build up down logs clean test init-dvc demo clearml minio

help:
	@echo "ML Service Commands:"
	@echo "  build        Build Docker images"
	@echo "  up           Start services (API + Dashboard + ClearML + MinIO)"
	@echo "  down         Stop services"
	@echo "  logs         Show logs"
	@echo "  clean        Remove containers and volumes"
	@echo "  test         Test all services"
	@echo "  api          Run API locally"
	@echo "  dashboard    Run dashboard locally"
	@echo "  init-dvc     Initialize DVC with datasets"
	@echo "  clearml      Open ClearML web interface"
	@echo "  minio        Open MinIO web interface"
	@echo "  status       Show service status and URLs"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	docker system prune -f

test:
	@echo "Testing services..."
	@echo "Testing REST API..."
	@powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:8000/api/v1/health' -UseBasicParsing | Out-Null; echo 'REST API is running' } catch { echo 'REST API is not running' }"
	@echo "Testing Dashboard..."
	@powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:7860' -UseBasicParsing | Out-Null; echo 'Dashboard is running' } catch { echo 'Dashboard is not running' }"
	@echo "Testing MinIO..."
	@powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:9001' -UseBasicParsing | Out-Null; echo 'MinIO is running' } catch { echo 'MinIO is not running' }"
	@echo "Testing ClearML..."
	@powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:8080' -UseBasicParsing | Out-Null; echo 'ClearML is running' } catch { echo 'ClearML is not running' }"

api:
	python api/main.py

dashboard:
	python dashboard/app.py

init-dvc:
	@echo "Initializing DVC with datasets..."
	python scripts/init_dvc.py

clearml:
	@echo "Opening ClearML web interface..."
	@powershell -Command "Start-Process 'http://localhost:8080'"

minio:
	@echo "Opening MinIO web interface..."
	@powershell -Command "Start-Process 'http://localhost:9001'"

status:
	@echo "Service URLs:"
	@echo "  REST API:    http://localhost:8000/docs"
	@echo "  Dashboard:   http://localhost:7860"
	@echo "  MinIO:       http://localhost:9001 (minioadmin/minioadmin)"
	@echo "  ClearML:     http://localhost:8080"
	@echo "  gRPC:        localhost:50051 (use python app/grpc_client.py)"

full-setup: init-dvc build up status
	@echo "Full setup completed! Services are starting..."
	@echo "Wait a moment and run 'make test' to verify everything is working"

.PHONY: all

all: build up status
