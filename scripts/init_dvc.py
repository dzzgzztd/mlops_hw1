import os
import subprocess


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def main() -> None:
    endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    bucket = os.getenv("DVC_S3_BUCKET", "dvc-storage")

    remote_url = f"s3://{bucket}"

    run(["dvc", "init", "--no-scm"])
    run(["dvc", "remote", "add", "-f", "minio", remote_url])
    run(["dvc", "remote", "modify", "minio", "endpointurl", endpoint_url])
    run(["dvc", "remote", "modify", "minio", "access_key_id", access_key])
    run(["dvc", "remote", "modify", "minio", "secret_access_key", secret_key])

    print("DVC remote configured:")
    print("  remote:", remote_url)
    print("  endpoint:", endpoint_url)


if __name__ == "__main__":
    main()