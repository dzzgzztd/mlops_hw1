.PHONY: help build up down logs clean test api dashboard grpc init-dvc clearml minio status full-setup \
        k8s-start k8s-build k8s-apply k8s-up k8s-status k8s-api-url k8s-dashboard-url k8s-down \
        k8s-port-api k8s-port-dashboard

COMPOSE = docker compose
K8S_NS = mlops-hw1

help:
	@echo "ML Service Commands:"
	@echo ""
	@echo "Docker Compose:"
	@echo "  make build            Build Docker images"
	@echo "  make up               Start local stack with Docker Compose"
	@echo "  make down             Stop local stack"
	@echo "  make logs             Show Docker Compose logs"
	@echo "  make clean            Stop stack and remove unused Docker resources"
	@echo "  make test             Check availability of local services"
	@echo "  make status           Show local service URLs"
	@echo "  make full-setup       Build + start local stack + show URLs"
	@echo ""
	@echo "Local run:"
	@echo "  make api              Run FastAPI locally"
	@echo "  make dashboard        Run Gradio dashboard locally"
	@echo "  make grpc             Run gRPC server locally"
	@echo "  make init-dvc         Initialize/configure DVC locally"
	@echo "  make clearml          Open ClearML UI"
	@echo "  make minio            Open MinIO UI"
	@echo ""
	@echo "Kubernetes / Minikube:"
	@echo "  make k8s-start        Start Minikube with Docker driver"
	@echo "  make k8s-build        Build images inside Minikube Docker daemon"
	@echo "  make k8s-apply        Apply Kubernetes manifests"
	@echo "  make k8s-up           Start Minikube + build images + apply manifests"
	@echo "  make k8s-status       Show pods, services and PVCs"
	@echo "  make k8s-api-url      Get API URL from Minikube service"
	@echo "  make k8s-dashboard-url Get dashboard URL from Minikube service"
	@echo "  make k8s-port-api     Port-forward API service to localhost:8000"
	@echo "  make k8s-port-dashboard Port-forward dashboard to localhost:7860"
	@echo "  make k8s-down         Delete Kubernetes namespace"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

clean:
	$(COMPOSE) down -v --remove-orphans
	docker system prune -f

test:
	@echo "Testing REST API..."
	@powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:8000/api/v1/health' -UseBasicParsing | Out-Null; echo 'REST API is running' } catch { echo 'REST API is not running' }"
	@echo "Testing Dashboard..."
	@powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:7860' -UseBasicParsing | Out-Null; echo 'Dashboard is running' } catch { echo 'Dashboard is not running' }"
	@echo "Testing MinIO..."
	@powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:9001' -UseBasicParsing | Out-Null; echo 'MinIO is running' } catch { echo 'MinIO is not running' }"
	@echo "Testing ClearML..."
	@powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:8080' -UseBasicParsing | Out-Null; echo 'ClearML is running' } catch { echo 'ClearML is not running' }"

api:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

dashboard:
	python dashboard/app.py

grpc:
	python -m app.grpc.service

init-dvc:
	python init_dvc.py

clearml:
	@powershell -Command "Start-Process 'http://localhost:8080'"

minio:
	@powershell -Command "Start-Process 'http://localhost:9001'"

status:
	@echo "Local service URLs:"
	@echo "  REST API:    http://localhost:8000/docs"
	@echo "  Dashboard:   http://localhost:7860"
	@echo "  MinIO:       http://localhost:9001"
	@echo "  ClearML:     http://localhost:8080"
	@echo "  gRPC:        localhost:50051"

full-setup: build up status

k8s-start:
	minikube start --driver=docker

k8s-build:
	powershell -Command "& minikube -p minikube docker-env --shell powershell | Invoke-Expression; docker build -t ml-api:latest -f Dockerfile .; docker build -t ml-dashboard:latest -f dashboard/Dockerfile ./dashboard"

k8s-apply:
	kubectl apply -f kubernetes/namespace.yaml
	kubectl apply -f kubernetes/secret.yaml
	kubectl apply -f kubernetes/configmap.yaml
	kubectl apply -f kubernetes/storage.yaml
	kubectl apply -f kubernetes/minio-deployment.yaml
	kubectl apply -f kubernetes/minio-service.yaml
	kubectl apply -f kubernetes/ml-api-deployment.yaml
	kubectl apply -f kubernetes/ml-api-service.yaml
	kubectl apply -f kubernetes/dashboard-deployment.yaml
	kubectl apply -f kubernetes/dashboard-service.yaml

k8s-up: k8s-start k8s-build k8s-apply

k8s-status:
	kubectl get pods -n $(K8S_NS)
	kubectl get svc -n $(K8S_NS)
	kubectl get pvc -n $(K8S_NS)

k8s-api-url:
	minikube service ml-api-service -n $(K8S_NS) --url

k8s-dashboard-url:
	minikube service ml-dashboard-service -n $(K8S_NS) --url

k8s-port-api:
	kubectl port-forward -n $(K8S_NS) service/ml-api-service 8000:8000

k8s-port-dashboard:
	kubectl port-forward -n $(K8S_NS) service/ml-dashboard-service 7860:7860

k8s-down:
	kubectl delete namespace $(K8S_NS) --ignore-not-found=true