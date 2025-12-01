from sklearn.datasets import make_classification

def generate_sample_data(n_samples: int = 100, n_features: int = 4) -> tuple:
    """Генерация тестовых данных для классификации"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )
    return X, y

SAMPLE_X, SAMPLE_Y = generate_sample_data()