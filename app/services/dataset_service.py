import logging
import os
import pandas as pd
from typing import Dict, List, Any
import subprocess

logger = logging.getLogger(__name__)


class DatasetService:
    """Сервис для работы с версионированными датасетами через DVC"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.available_datasets = self._discover_datasets()

    def _discover_datasets(self) -> List[Dict[str, Any]]:
        """Обнаружение датасетов с DVC метаданными"""
        datasets = []

        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return datasets

        for item in os.listdir(self.data_dir):
            if item.endswith('.csv'):
                dataset_path = os.path.join(self.data_dir, item)
                dvc_path = dataset_path + '.dvc'

                dataset_info = {
                    'name': item.replace('.csv', ''),
                    'path': dataset_path,
                    'dvc_tracked': os.path.exists(dvc_path),
                    'size': os.path.getsize(dataset_path) if os.path.exists(dataset_path) else 0
                }

                # Получаем информацию о версии из DVC
                if dataset_info['dvc_tracked']:
                    try:
                        result = subprocess.run(
                            ['dvc', 'status', dataset_path],
                            capture_output=True, text=True
                        )
                        dataset_info['status'] = 'committed' if 'not in cache' not in result.stdout else 'modified'
                    except Exception as e:
                        logger.error(f"Error checking DVC status for {item}: {e}")
                        dataset_info['status'] = 'unknown'

                datasets.append(dataset_info)

        logger.info(f"Discovered {len(datasets)} datasets")
        return datasets

    def list_datasets(self) -> Dict[str, Any]:
        """Список доступных датасетов с метаданными"""
        try:
            datasets_info = []

            for dataset in self.available_datasets:
                info = {
                    'name': dataset['name'],
                    'dvc_tracked': dataset['dvc_tracked'],
                    'status': dataset.get('status', 'not_tracked'),
                    'size_mb': round(dataset['size'] / 1024 / 1024, 2) if dataset['size'] > 0 else 0
                }

                # Добавляем базовую статистику для CSV файлов
                if os.path.exists(dataset['path']):
                    try:
                        df = pd.read_csv(dataset['path'])
                        info.update({
                            'rows': len(df),
                            'columns': len(df.columns),
                            'columns_list': df.columns.tolist()
                        })
                    except Exception as e:
                        logger.warning(f"Could not read dataset {dataset['name']}: {e}")
                        info['error'] = str(e)

                datasets_info.append(info)

            return {
                'status': 'success',
                'datasets': datasets_info,
                'count': len(datasets_info)
            }

        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def get_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Загрузка конкретного датасета"""
        try:
            dataset_path = os.path.join(self.data_dir, f"{dataset_name}.csv")

            if not os.path.exists(dataset_path):
                return {
                    'status': 'error',
                    'error': f"Dataset {dataset_name} not found"
                }

            # Если файл отслеживается DVC, проверяем актуальность
            dvc_path = dataset_path + '.dvc'
            if os.path.exists(dvc_path):
                try:
                    subprocess.run(['dvc', 'pull', dataset_path], check=True)
                    logger.info(f"Pulled latest version of {dataset_name} from DVC")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Could not pull {dataset_name} from DVC: {e}")

            # Загружаем данные
            df = pd.read_csv(dataset_path)

            return {
                'status': 'success',
                'name': dataset_name,
                'data': df.to_dict('records'),
                'columns': df.columns.tolist(),
                'shape': df.shape,
                'description': f"Dataset {dataset_name} with {len(df)} rows and {len(df.columns)} columns"
            }

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def pull_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Принудительное обновление датасета из DVC"""
        try:
            dataset_path = os.path.join(self.data_dir, f"{dataset_name}.csv")
            dvc_path = dataset_path + '.dvc'

            if not os.path.exists(dvc_path):
                return {
                    'status': 'error',
                    'error': f"Dataset {dataset_name} is not tracked by DVC"
                }

            subprocess.run(['dvc', 'pull', dataset_path], check=True)
            logger.info(f"Successfully pulled {dataset_name} from DVC")

            return {
                'status': 'success',
                'message': f"Dataset {dataset_name} updated from DVC"
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Error pulling dataset {dataset_name}: {e}")
            return {
                'status': 'error',
                'error': f"DVC pull failed: {e}"
            }

    def add_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Добавление нового датасета в DVC"""
        try:
            if not os.path.exists(dataset_path):
                return {
                    'status': 'error',
                    'error': f"File {dataset_path} does not exist"
                }

            # Копируем файл
            import shutil
            filename = os.path.basename(dataset_path)
            target_path = os.path.join(self.data_dir, filename)

            shutil.copy2(dataset_path, target_path)

            # Добавляем в DVC
            subprocess.run(['dvc', 'add', target_path], check=True)
            subprocess.run(['dvc', 'push'], check=True)

            logger.info(f"Successfully added {filename} to DVC")

            # Обновляем список датасетов
            self.available_datasets = self._discover_datasets()

            return {
                'status': 'success',
                'message': f"Dataset {filename} added and tracked by DVC"
            }

        except Exception as e:
            logger.error(f"Error adding dataset {dataset_path}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }