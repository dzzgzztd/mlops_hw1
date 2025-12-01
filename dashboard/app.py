import gradio as gr
import requests
import json
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000/api/v1"


class MLDashboard:
    """Класс для управления дашбордом"""

    def __init__(self, api_base_url: str = API_BASE_URL):
        self.api_base_url = api_base_url
        self.current_models = []
        self.available_models = {}
        self.datasets = []

    def _make_api_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Вспомогательный метод для API запросов"""
        url = f"{self.api_base_url}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                response = requests.post(url, json=data)
            elif method == "PUT":
                response = requests.put(url, json=data)
            elif method == "DELETE":
                response = requests.delete(url)
            else:
                return {"status": "error", "error": f"Unsupported method: {method}"}

            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}: {response.text}"}

        except requests.exceptions.ConnectionError:
            return {"status": "error",
                    "error": "Cannot connect to API server. Make sure it's running on localhost:8000"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def health_check(self) -> str:
        """Проверка статуса сервиса"""
        result = self._make_api_request("/health")
        if result.get("status") == "healthy":
            return "Сервис работает нормально"
        else:
            return f"Ошибка: {result.get('error', 'Unknown error')}"

    def load_available_models(self) -> Dict:
        """Загрузка списка доступных моделей"""
        result = self._make_api_request("/models/available")
        if result.get("status") == "success":
            self.available_models = result.get("available_models", {})
            return result
        return result

    def load_trained_models(self) -> Dict:
        """Загрузка списка обученных моделей"""
        result = self._make_api_request("/models")
        if result.get("status") == "success":
            self.current_models = result.get("models", [])
        return result

    def load_datasets(self) -> Dict:
        """Загрузка списка датасетов"""
        result = self._make_api_request("/datasets")
        if result.get("status") == "success":
            self.datasets = result.get("datasets", [])
        return result

    def train_model_interface(self, model_type: str, dataset_name: str, hyperparameters_json: str) -> str:
        """Интерфейс для обучения модели с выбором датасета"""
        try:
            # Парсим JSON с гиперпараметрами
            if hyperparameters_json.strip():
                hyperparameters = json.loads(hyperparameters_json)
            else:
                hyperparameters = {}

            # Отправляем запрос на обучение
            result = self._make_api_request(
                "/models/train",
                "POST",
                {
                    "model_type": model_type,
                    "dataset_name": dataset_name,
                    "hyperparameters": hyperparameters
                }
            )

            if result.get("status") == "success":
                model_id = result.get("model_id")
                accuracy = result.get("train_accuracy", 0)
                clearml_id = result.get("clearml_task_id")
                dataset_info = result.get("dataset_info", "")

                response = f"Модель успешно обучена!\n"
                response += f"ID: {model_id}\n"
                response += f"Accuracy: {accuracy:.4f}\n"
                response += f"Датасет: {dataset_info}\n"
                if clearml_id:
                    response += f"ClearML Task: {clearml_id}\n"
                return response
            else:
                return f"Ошибка обучения: {result.get('error', 'Unknown error')}"

        except json.JSONDecodeError as e:
            return f"Ошибка в формате JSON: {str(e)}"
        except Exception as e:
            return f"Неожиданная ошибка: {str(e)}"

    def predict_interface(self, model_id: str, input_data_json: str) -> str:
        """Интерфейс для предсказания"""
        try:
            # Парсим JSON с данными для предсказания
            if not input_data_json.strip():
                return "Введите данные для предсказания в формате JSON"

            input_data = json.loads(input_data_json)

            # Проверяем формат данных
            if not isinstance(input_data, list) or not all(isinstance(row, list) for row in input_data):
                return "Данные должны быть списком списков (2D массив)"

            # Отправляем запрос на предсказание
            result = self._make_api_request(
                f"/models/{model_id}/predict",
                "POST",
                {"data": input_data}
            )

            if result.get("status") == "success":
                predictions = result.get("predictions", [])
                return f"Предсказания: {predictions}"
            else:
                return f"Ошибка предсказания: {result.get('error', 'Unknown error')}"

        except json.JSONDecodeError as e:
            return f"Ошибка в формате JSON: {str(e)}"
        except Exception as e:
            return f"Неожиданная ошибка: {str(e)}"

    def delete_model_interface(self, model_id: str) -> str:
        """Интерфейс для удаления модели"""
        if not model_id:
            return "Выберите модель для удаления"

        result = self._make_api_request(f"/models/{model_id}", "DELETE")

        if result.get("status") == "success":
            return f"Модель {model_id} успешно удалена"
        else:
            return f"Ошибка удаления: {result.get('error', 'Unknown error')}"

    def retrain_model_interface(self, model_id: str, dataset_name: str, hyperparameters_json: str) -> str:
        """Интерфейс для переобучения модели с выбором датасета"""
        try:
            if not model_id:
                return "Выберите модель для переобучения"

            # Парсим JSON с гиперпараметрами
            if hyperparameters_json.strip():
                hyperparameters = json.loads(hyperparameters_json)
            else:
                hyperparameters = {}

            # Отправляем запрос на переобучение
            result = self._make_api_request(
                f"/models/{model_id}/retrain",
                "PUT",
                {
                    "dataset_name": dataset_name,
                    "hyperparameters": hyperparameters
                }
            )

            if result.get("status") == "success":
                accuracy = result.get("train_accuracy", 0)
                clearml_id = result.get("clearml_task_id")

                response = f"Модель успешно переобучена!\n"
                response += f"Accuracy: {accuracy:.4f}\n"
                if clearml_id:
                    response += f"ClearML Task: {clearml_id}\n"
                return response
            else:
                return f"Ошибка переобучения: {result.get('error', 'Unknown error')}"

        except json.JSONDecodeError as e:
            return f"Ошибка в формате JSON: {str(e)}"
        except Exception as e:
            return f"Неожиданная ошибка: {str(e)}"

    def get_model_info_interface(self, model_id: str) -> str:
        """Информация о модели"""
        if not model_id:
            return "Выберите модель для просмотра информации"

        result = self._make_api_request(f"/models/{model_id}")

        if result.get("status") == "success":
            info = f"ID: {result.get('model_id')}\n"
            info += f"Тип: {result.get('model_type')}\n"
            info += f"Обучена: {'Да' if result.get('is_trained') else 'Нет'}"
            return info
        else:
            return f"Ошибка: {result.get('error', 'Unknown error')}"

    def pull_dataset_interface(self, dataset_name: str) -> str:
        """Обновление датасета из DVC"""
        if not dataset_name:
            return "Выберите датасет для обновления"

        result = self._make_api_request(f"/datasets/{dataset_name}/pull", "POST")

        if result.get("status") == "success":
            return f"Датасет {dataset_name} обновлен из DVC"
        else:
            return f"Ошибка обновления: {result.get('error', 'Unknown error')}"

    def get_dataset_info_interface(self, dataset_name: str) -> str:
        """Информация о датасете"""
        if not dataset_name:
            return "Выберите датасет для просмотра информации"

        result = self._make_api_request(f"/datasets/{dataset_name}")

        if result.get("status") == "success":
            info = f"Датасет: {result.get('name')}\n"
            info += f"Размер: {result.get('shape', [0, 0])[0]} строк, {result.get('shape', [0, 0])[1]} колонок\n"
            info += f"Колонки: {', '.join(result.get('columns', []))}\n"
            info += f"Описание: {result.get('description', '')}"
            return info
        else:
            return f"Ошибка: {result.get('error', 'Unknown error')}"

    def create_dashboard(self) -> gr.Blocks:
        """Создание интерфейса"""

        with gr.Blocks(title="ML Model Dashboard") as dashboard:
            gr.Markdown("# ML Model Dashboard")
            gr.Markdown("Дашборд для управления ML моделями через REST API с DVC и ClearML")

            # Вкладки
            with gr.Tab("Обзор"):
                with gr.Row():
                    with gr.Column():
                        status_btn = gr.Button("Проверить статус")
                        status_output = gr.Textbox(label="Статус сервиса", interactive=False)

                    with gr.Column():
                        refresh_btn = gr.Button("Обновить данные")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Статистика")
                        models_count = gr.Textbox(label="Количество обученных моделей", interactive=False)
                        datasets_count = gr.Textbox(label="Доступные датасеты", interactive=False)

            with gr.Tab("Обучение моделей"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Выбор модели и параметров")
                        model_dropdown = gr.Dropdown(
                            choices=[],
                            label="Тип модели",
                            info="Выберите тип модели для обучения"
                        )
                        dataset_dropdown_train = gr.Dropdown(
                            choices=[],
                            label="Датасет для обучения",
                            info="Выберите датасет из DVC"
                        )
                        hyperparameters_json = gr.Textbox(
                            label="Гиперпараметры (JSON)",
                            placeholder='{"C": 0.1, "max_iter": 100}',
                            lines=3
                        )
                        train_btn = gr.Button("Обучить модель", variant="primary")

                    with gr.Column():
                        gr.Markdown("### Доступные модели")
                        available_models_json = gr.JSON(
                            label="Информация о моделях"
                        )

                with gr.Row():
                    train_output = gr.Textbox(
                        label="Результат обучения",
                        interactive=False,
                        lines=6
                    )

            with gr.Tab("Предсказание"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Выбор модели и данных")
                        trained_models_dropdown = gr.Dropdown(
                            choices=[],
                            label="Обученные модели",
                            info="Выберите модель для предсказания"
                        )
                        predict_input = gr.Textbox(
                            label="Данные для предсказания (JSON)",
                            placeholder='[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]',
                            lines=4
                        )
                        predict_btn = gr.Button("Получить предсказание", variant="primary")

                    with gr.Column():
                        gr.Markdown("### Информация о модели")
                        model_info_btn = gr.Button("Получить информацию")
                        model_info_output = gr.Textbox(
                            label="Информация о модели",
                            interactive=False,
                            lines=5
                        )

                with gr.Row():
                    predict_output = gr.Textbox(
                        label="Результат предсказания",
                        interactive=False,
                        lines=3
                    )

            with gr.Tab("Управление моделями"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Переобучение модели")
                        retrain_model_dropdown = gr.Dropdown(
                            choices=[],
                            label="Модель для переобучения"
                        )
                        dataset_dropdown_retrain = gr.Dropdown(
                            choices=[],
                            label="Датасет для переобучения",
                            info="Выберите датасет из DVC"
                        )
                        retrain_hyperparams = gr.Textbox(
                            label="Новые гиперпараметры (JSON)",
                            placeholder='{"C": 0.5, "max_iter": 200}',
                            lines=2
                        )
                        retrain_btn = gr.Button("Переобучить модель")
                        retrain_output = gr.Textbox(label="Результат", interactive=False, lines=4)

                    with gr.Column():
                        gr.Markdown("### Удаление модели")
                        delete_model_dropdown = gr.Dropdown(
                            choices=[],
                            label="Модель для удаления"
                        )
                        delete_btn = gr.Button("Удалить модель", variant="stop")
                        delete_output = gr.Textbox(label="Результат", interactive=False)

            with gr.Tab("Датасеты"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Доступные датасеты")
                        datasets_display = gr.JSON(
                            label="Список датасетов с DVC информацией"
                        )
                        refresh_datasets_btn = gr.Button("Обновить список")

                    with gr.Column():
                        gr.Markdown("### Управление датасетами")
                        dataset_info_dropdown = gr.Dropdown(
                            choices=[],
                            label="Выберите датасет"
                        )
                        with gr.Row():
                            dataset_info_btn = gr.Button("Информация о датасете")
                            dataset_pull_btn = gr.Button("Обновить из DVC")
                        dataset_info_output = gr.Textbox(
                            label="Информация о датасете",
                            interactive=False,
                            lines=6
                        )
                        gr.Markdown("""
                        **DVC Интеграция:**
                        - Датасеты версионируются через DVC
                        - Хранятся в MinIO (S3)
                        - Автоматическое обновление
                        """)

            # Обработчики событий
            def refresh_all_data():
                """Обновление всех данных"""
                # Загружаем доступные модели
                models_result = self.load_available_models()
                model_choices = list(models_result.get("available_models", {}).keys()) if models_result.get(
                    "status") == "success" else []

                # Загружаем обученные модели
                trained_result = self.load_trained_models()
                trained_choices = [model["model_id"] for model in
                                   trained_result.get("models", [])] if trained_result.get(
                    "status") == "success" else []

                # Загружаем датасеты
                datasets_result = self.load_datasets()
                dataset_names = [dataset.get('name', '') for dataset in
                                 datasets_result.get('datasets', [])] if datasets_result.get(
                    'status') == 'success' else []

                # Статистика
                models_count_text = f"{len(trained_choices)} моделей"
                datasets_count_text = f"{len(dataset_names)} датасетов"

                return {
                    model_dropdown: gr.update(choices=model_choices),
                    trained_models_dropdown: gr.update(choices=trained_choices),
                    retrain_model_dropdown: gr.update(choices=trained_choices),
                    delete_model_dropdown: gr.update(choices=trained_choices),
                    dataset_dropdown_train: gr.update(choices=dataset_names),
                    dataset_dropdown_retrain: gr.update(choices=dataset_names),
                    dataset_info_dropdown: gr.update(choices=dataset_names),
                    available_models_json: models_result.get("available_models", {}),
                    datasets_display: datasets_result.get("datasets", []),
                    models_count: models_count_text,
                    datasets_count: datasets_count_text
                }

            # Привязка обработчиков
            status_btn.click(
                fn=self.health_check,
                outputs=status_output
            )

            refresh_btn.click(
                fn=refresh_all_data,
                outputs=[
                    model_dropdown, trained_models_dropdown, retrain_model_dropdown,
                    delete_model_dropdown, dataset_dropdown_train, dataset_dropdown_retrain,
                    dataset_info_dropdown, available_models_json, datasets_display,
                    models_count, datasets_count
                ]
            )

            refresh_datasets_btn.click(
                fn=lambda: self.load_datasets(),
                outputs=datasets_display
            )

            train_btn.click(
                fn=self.train_model_interface,
                inputs=[model_dropdown, dataset_dropdown_train, hyperparameters_json],
                outputs=train_output
            )

            predict_btn.click(
                fn=self.predict_interface,
                inputs=[trained_models_dropdown, predict_input],
                outputs=predict_output
            )

            model_info_btn.click(
                fn=self.get_model_info_interface,
                inputs=[trained_models_dropdown],
                outputs=model_info_output
            )

            retrain_btn.click(
                fn=self.retrain_model_interface,
                inputs=[retrain_model_dropdown, dataset_dropdown_retrain, retrain_hyperparams],
                outputs=retrain_output
            )

            delete_btn.click(
                fn=self.delete_model_interface,
                inputs=[delete_model_dropdown],
                outputs=delete_output
            )

            dataset_info_btn.click(
                fn=self.get_dataset_info_interface,
                inputs=[dataset_info_dropdown],
                outputs=dataset_info_output
            )

            dataset_pull_btn.click(
                fn=self.pull_dataset_interface,
                inputs=[dataset_info_dropdown],
                outputs=dataset_info_output
            )

            # Инициализация данных при загрузке
            dashboard.load(
                fn=refresh_all_data,
                outputs=[
                    model_dropdown, trained_models_dropdown, retrain_model_dropdown,
                    delete_model_dropdown, dataset_dropdown_train, dataset_dropdown_retrain,
                    dataset_info_dropdown, available_models_json, datasets_display,
                    models_count, datasets_count
                ]
            )

        return dashboard


def main():
    """Запуск дашборда"""
    dashboard = MLDashboard()
    app = dashboard.create_dashboard()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
