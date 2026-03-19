import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from app.core.exceptions import DatasetNotFoundError, DatasetLoadError

logger = logging.getLogger(__name__)


class DatasetService:
    """Сервис для работы с версионированными датасетами через DVC"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def _discover_datasets(self) -> List[Dict[str, Any]]:
        """Обнаружение датасетов по .dvc-файлам и/или csv"""
        datasets: List[Dict[str, Any]] = []

        if not self.data_dir.exists():
            logger.warning("Data directory %s does not exist", self.data_dir)
            return datasets

        seen = set()

        # Основной источник истины — .csv.dvc
        for dvc_file in self.data_dir.glob("*.csv.dvc"):
            dataset_name = dvc_file.name.replace(".csv.dvc", "")
            csv_path = self.data_dir / f"{dataset_name}.csv"
            seen.add(dataset_name)

            dataset_info = {
                "name": dataset_name,
                "path": str(csv_path),
                "dvc_file": str(dvc_file),
                "dvc_tracked": True,
                "local_exists": csv_path.exists(),
                "size": csv_path.stat().st_size if csv_path.exists() else 0,
            }

            datasets.append(dataset_info)

        # На случай локальных csv, которые ещё не добавлены в DVC
        for csv_file in self.data_dir.glob("*.csv"):
            dataset_name = csv_file.stem
            if dataset_name in seen:
                continue

            datasets.append(
                {
                    "name": dataset_name,
                    "path": str(csv_file),
                    "dvc_file": str(csv_file) + ".dvc",
                    "dvc_tracked": False,
                    "local_exists": True,
                    "size": csv_file.stat().st_size,
                }
            )

        logger.info("Discovered %s datasets", len(datasets))
        return datasets

    def list_datasets(self) -> Dict[str, Any]:
        """Список доступных датасетов с метаданными"""
        try:
            datasets = self._discover_datasets()
            datasets_info = []

            for dataset in datasets:
                info = {
                    "name": dataset["name"],
                    "dvc_tracked": dataset["dvc_tracked"],
                    "status": "tracked" if dataset["dvc_tracked"] else "local_only",
                    "local_exists": dataset["local_exists"],
                    "size_mb": round(dataset["size"] / 1024 / 1024, 4) if dataset["size"] > 0 else 0,
                }

                csv_path = Path(dataset["path"])
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        info.update(
                            {
                                "rows": len(df),
                                "columns": len(df.columns),
                                "columns_list": df.columns.tolist(),
                            }
                        )
                    except Exception as e:
                        logger.warning("Could not read dataset %s: %s", dataset["name"], e)
                        info["error"] = str(e)

                datasets_info.append(info)

            return {
                "status": "success",
                "datasets": datasets_info,
                "count": len(datasets_info),
            }

        except Exception as e:
            logger.exception("Error listing datasets: %s", e)
            return {
                "status": "error",
                "error": str(e),
            }

    def get_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Загрузка конкретного датасета"""
        try:
            dataset_path = self.data_dir / f"{dataset_name}.csv"
            dvc_path = self.data_dir / f"{dataset_name}.csv.dvc"

            # Если файл отсутствует локально, но есть DVC-метаданные — тянем из remote
            if not dataset_path.exists() and dvc_path.exists():
                try:
                    subprocess.run(
                        ["dvc", "pull", str(dvc_path)],
                        check=True,
                        cwd="/app",
                        capture_output=True,
                        text=True,
                    )
                    logger.info("Pulled %s from DVC", dataset_name)
                except subprocess.CalledProcessError as e:
                    logger.warning("Could not pull %s from DVC: %s", dataset_name, e.stderr)

            if not dataset_path.exists():
                return {
                    "status": "error",
                    "error": f"Dataset {dataset_name} not found",
                }

            df = pd.read_csv(dataset_path)

            return {
                "status": "success",
                "name": dataset_name,
                "data": df.to_dict("records"),
                "columns": df.columns.tolist(),
                "shape": list(df.shape),
                "description": f"Dataset {dataset_name} with {len(df)} rows and {len(df.columns)} columns",
            }

        except Exception as e:
            logger.exception("Error loading dataset %s: %s", dataset_name, e)
            return {
                "status": "error",
                "error": str(e),
            }

    def pull_dataset(self, dataset_name: str):
        dvc_file = self.data_dir / f"{dataset_name}.csv.dvc"

        if not dvc_file.exists():
            raise DatasetNotFoundError(f"Датасет {dataset_name} не отслеживается через DVC")

        try:
            subprocess.run(
                ["dvc", "pull", str(dvc_file)],
                cwd="/app",
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("DVC pull выполнен для %s", dataset_name)
            return {
                "message": f"Датасет {dataset_name} обновлён из DVC"
            }
        except subprocess.CalledProcessError as e:
            raise DatasetLoadError(
                f"Ошибка DVC pull для {dataset_name}: {e.stderr or str(e)}"
            ) from e

    def add_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Добавление нового датасета в DVC"""
        try:
            src = Path(dataset_path)
            if not src.exists():
                return {
                    "status": "error",
                    "error": f"File {dataset_path} does not exist",
                }

            filename = src.name
            target_path = self.data_dir / filename
            shutil.copy2(src, target_path)

            subprocess.run(["dvc", "add", str(target_path)], check=True, cwd="/app")
            subprocess.run(["dvc", "push"], check=True, cwd="/app")

            logger.info("Successfully added %s to DVC", filename)

            return {
                "status": "success",
                "message": f"Dataset {filename} added and tracked by DVC",
            }

        except Exception as e:
            logger.exception("Error adding dataset %s: %s", dataset_path, e)
            return {
                "status": "error",
                "error": str(e),
            }
