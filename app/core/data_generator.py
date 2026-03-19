from functools import lru_cache
from sklearn.datasets import make_classification


def generate_sample_data(
    n_samples: int = 100,
    n_features: int = 4,
    n_classes: int = 2,
    random_state: int = 42,
):
    """Генерация данных для классификации"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_redundant=1,
        n_classes=n_classes,
        random_state=random_state,
    )
    return X, y


@lru_cache(maxsize=1)
def get_sample_data():
    return generate_sample_data()