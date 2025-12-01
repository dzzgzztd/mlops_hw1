# scripts/init_dvc.py
import os
import subprocess
import sys
import time


def create_minio_bucket():
    """Создаем bucket в MinIO если его нет"""
    try:
        import boto3
        from botocore.exceptions import ClientError

        # Создаем клиент для MinIO
        s3_client = boto3.client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin',
            region_name='us-east-1'
        )

        # Пробуем создать bucket
        try:
            s3_client.create_bucket(Bucket='dvc-storage')
            print("Created bucket 'dvc-storage' in MinIO")
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                print("Bucket 'dvc-storage' already exists")
            else:
                print(f"Could not create bucket: {e}")

    except ImportError:
        print("boto3 not available, trying alternative method...")
        try:
            import requests
            # Создаем bucket через MinIO API
            response = requests.put(
                'http://localhost:9000/dvc-storage',
                auth=('minioadmin', 'minioadmin')
            )
            if response.status_code in [200, 409]:  # 409 = уже существует
                print("Bucket 'dvc-storage' created/exists in MinIO")
            else:
                print(f"Could not create bucket via API: {response.status_code}")
        except:
            print("Could not create bucket automatically")
            print("Please create bucket manually in MinIO UI: http://localhost:9001")
            return False
    return True


def is_minio_running():
    """Проверяем, запущен ли MinIO"""
    try:
        import requests
        response = requests.get("http://localhost:9001", timeout=5)
        return response.status_code == 200
    except:
        return False


def init_dvc():
    """Инициализация DVC с MinIO как remote storage"""

    print("Initializing DVC...")

    # Проверяем, запущен ли MinIO
    if not is_minio_running():
        print("   MinIO is not running. Please start it first:")
        print("   docker-compose up minio -d")
        print("   or run: make up")
        sys.exit(1)

    # Создаем bucket в MinIO
    if not create_minio_bucket():
        print("   Please create bucket 'dvc-storage' manually in MinIO UI")
        print("   Open: http://localhost:9001")
        print("   Login: minioadmin / minioadmin")
        print("   Click 'Create Bucket' and name it 'dvc-storage'")
        response = input("Press Enter after creating the bucket, or Ctrl+C to cancel...")

    try:
        # Инициализируем DVC
        subprocess.run(["dvc", "init", "--no-scm", "-f"], check=True)
        print("DVC initialized")

        # Настраиваем remote storage
        config_commands = [
            ["dvc", "remote", "add", "-d", "myremote", "s3://dvc-storage"],
            ["dvc", "remote", "modify", "myremote", "endpointurl", "http://localhost:9000"],
            ["dvc", "remote", "modify", "myremote", "access_key_id", "minioadmin"],
            ["dvc", "remote", "modify", "myremote", "secret_access_key", "minioadmin"]
        ]

        for cmd in config_commands:
            subprocess.run(cmd, check=True)

        print("MinIO configured as DVC remote")

        # Добавляем датасеты в DVC
        print("Adding datasets to DVC...")
        datasets = ["data/iris.csv", "data/wine.csv"]

        for dataset in datasets:
            if os.path.exists(dataset):
                subprocess.run(["dvc", "add", dataset], check=True)
                print(f"Added {dataset} to DVC")

        print("Waiting for MinIO to be ready...")
        time.sleep(3)

        # Пушим данные
        print("Pushing data to MinIO...")
        subprocess.run(["dvc", "push"], check=True)

        print("DVC initialization completed!")
        print("Datasets are now versioned and stored in MinIO")

    except subprocess.CalledProcessError as e:
        print(f"Error during DVC initialization: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr.decode()}")
        sys.exit(1)


if __name__ == "__main__":
    init_dvc()