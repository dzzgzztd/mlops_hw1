import time
import socket


def check_port(host, port, timeout=5):
    """Проверяем доступность порта"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        return result == 0
    except:
        return False
    finally:
        sock.close()


def wait_for_service(name, host, port, timeout=120):
    """Ждем пока сервис станет доступен"""
    print(f"Waiting for {name} ({host}:{port})...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if check_port(host, port, 2):
            print(f"{name} is available!")
            return True

        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0:
            print(f"   Still waiting... ({elapsed}s/{timeout}s)")
        time.sleep(2)

    print(f"Timeout waiting for {name}")
    return False


def main():
    """Ожидание всех необходимых сервисов"""

    print("Starting service health check...")

    services = [
        ("MongoDB", "localhost", 27017),
        ("Redis", "localhost", 6379),
        ("Elasticsearch", "localhost", 9200),
        ("ClearML API", "localhost", 8008),
        ("ClearML Web", "localhost", 8080),
        ("ClearML Files", "localhost", 8081),
    ]

    all_ready = True
    for name, host, port in services:
        if not wait_for_service(name, host, port, 90):
            all_ready = False
            print(f"{name} may not be fully operational")

    if all_ready:
        print("\n All services are ready!")
        print("   ClearML UI: http://localhost:8080")
        print("   ML API:     http://localhost:8000/docs")
        print("   Dashboard:  http://localhost:7860")
    else:
        print("\n Some services may need attention")
        print("Check logs with: docker-compose logs [service_name]")


if __name__ == "__main__":
    main()
