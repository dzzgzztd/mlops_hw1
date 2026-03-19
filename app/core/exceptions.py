class MLServiceError(Exception):
    """Базовое исключение сервиса."""
    pass


class ModelNotFoundError(MLServiceError):
    """Модель не найдена."""
    pass


class DatasetNotFoundError(MLServiceError):
    """Датасет не найден."""
    pass


class DatasetLoadError(MLServiceError):
    """Ошибка загрузки датасета."""
    pass


class TrainingError(MLServiceError):
    """Ошибка обучения модели."""
    pass


class PredictionError(MLServiceError):
    """Ошибка инференса."""
    pass


class ModelDeleteError(MLServiceError):
    """Ошибка удаления модели."""
    pass


class ClearMLError(MLServiceError):
    """Ошибка интеграции с ClearML."""
    pass