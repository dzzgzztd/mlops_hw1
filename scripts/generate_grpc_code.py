import subprocess
import sys
import os


def generate_grpc_stubs():
    """Генерация кода из .proto файлов"""

    proto_dir = "app/protos"
    output_dir = "app/grpc/generated"

    os.makedirs(output_dir, exist_ok=True)

    command = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        f"--pyi_out={output_dir}",
        f"{proto_dir}/ml_service.proto"
    ]

    try:
        print("Генерация gRPC кода...")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("gRPC код успешно сгенерирован!")

        # Исправляем импорты в сгенерированных файлах
        fix_imports(output_dir)

    except subprocess.CalledProcessError as e:
        print(f"Ошибка генерации gRPC кода: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        sys.exit(1)


def fix_imports(output_dir: str):
    """Исправление импортов в сгенерированных файлах"""
    files_to_fix = [
        "ml_service_pb2.py",
        "ml_service_pb2_grpc.py",
        "ml_service_pb2.pyi"
    ]

    for filename in files_to_fix:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Заменяем абсолютные импорты на относительные
            content = content.replace(
                "import ml_service_pb2 as ml__service__pb2",
                "from . import ml_service_pb2 as ml__service__pb2"
            )

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"Исправлены импорты в {filename}")


if __name__ == "__main__":
    generate_grpc_stubs()