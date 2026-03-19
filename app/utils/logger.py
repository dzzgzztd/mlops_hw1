import logging
import sys
from pathlib import Path


def setup_logger(name: str = "ml_service", log_level: str = "INFO") -> logging.Logger:
    """Настройка логгера"""

    # Определяем путь к корню проекта
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    level = getattr(logging, log_level.upper())
    logger.setLevel(level)

    # Проверяем, нет ли уже обработчиков
    if not logger.handlers:
        # Форматтер
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Обработчик для stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)

        # Обработчик для файла
        log_path = log_dir / "ml_service.log"

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        # Добавляем обработчики к логгеру
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger