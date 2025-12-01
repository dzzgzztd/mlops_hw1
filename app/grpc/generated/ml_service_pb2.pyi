from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("status", "message", "timestamp")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    status: str
    message: str
    timestamp: str
    def __init__(self, status: _Optional[str] = ..., message: _Optional[str] = ..., timestamp: _Optional[str] = ...) -> None: ...

class AvailableModelsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ModelHyperparameters(_message.Message):
    __slots__ = ("parameters",)
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    parameters: _containers.ScalarMap[str, str]
    def __init__(self, parameters: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ModelInfo(_message.Message):
    __slots__ = ("model_type", "default_hyperparameters", "description")
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_HYPERPARAMETERS_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    model_type: str
    default_hyperparameters: ModelHyperparameters
    description: str
    def __init__(self, model_type: _Optional[str] = ..., default_hyperparameters: _Optional[_Union[ModelHyperparameters, _Mapping]] = ..., description: _Optional[str] = ...) -> None: ...

class AvailableModelsResponse(_message.Message):
    __slots__ = ("status", "available_models", "error")
    class AvailableModelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ModelInfo
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ModelInfo, _Mapping]] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_MODELS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: str
    available_models: _containers.MessageMap[str, ModelInfo]
    error: str
    def __init__(self, status: _Optional[str] = ..., available_models: _Optional[_Mapping[str, ModelInfo]] = ..., error: _Optional[str] = ...) -> None: ...

class TrainModelRequest(_message.Message):
    __slots__ = ("model_type", "hyperparameters")
    class HyperparametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    HYPERPARAMETERS_FIELD_NUMBER: _ClassVar[int]
    model_type: str
    hyperparameters: _containers.ScalarMap[str, str]
    def __init__(self, model_type: _Optional[str] = ..., hyperparameters: _Optional[_Mapping[str, str]] = ...) -> None: ...

class TrainModelResponse(_message.Message):
    __slots__ = ("status", "model_id", "model_type", "hyperparameters", "train_accuracy", "model_path", "error")
    class HyperparametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    HYPERPARAMETERS_FIELD_NUMBER: _ClassVar[int]
    TRAIN_ACCURACY_FIELD_NUMBER: _ClassVar[int]
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: str
    model_id: str
    model_type: str
    hyperparameters: _containers.ScalarMap[str, str]
    train_accuracy: float
    model_path: str
    error: str
    def __init__(self, status: _Optional[str] = ..., model_id: _Optional[str] = ..., model_type: _Optional[str] = ..., hyperparameters: _Optional[_Mapping[str, str]] = ..., train_accuracy: _Optional[float] = ..., model_path: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class ListModelsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ModelSummary(_message.Message):
    __slots__ = ("model_id", "model_type", "is_trained")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    IS_TRAINED_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    model_type: str
    is_trained: bool
    def __init__(self, model_id: _Optional[str] = ..., model_type: _Optional[str] = ..., is_trained: bool = ...) -> None: ...

class ListModelsResponse(_message.Message):
    __slots__ = ("status", "models", "count", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: str
    models: _containers.RepeatedCompositeFieldContainer[ModelSummary]
    count: int
    error: str
    def __init__(self, status: _Optional[str] = ..., models: _Optional[_Iterable[_Union[ModelSummary, _Mapping]]] = ..., count: _Optional[int] = ..., error: _Optional[str] = ...) -> None: ...

class ModelInfoRequest(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class ModelInfoResponse(_message.Message):
    __slots__ = ("status", "model_id", "model_type", "is_trained", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    IS_TRAINED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: str
    model_id: str
    model_type: str
    is_trained: bool
    error: str
    def __init__(self, status: _Optional[str] = ..., model_id: _Optional[str] = ..., model_type: _Optional[str] = ..., is_trained: bool = ..., error: _Optional[str] = ...) -> None: ...

class PredictRequest(_message.Message):
    __slots__ = ("model_id", "data")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    data: _containers.RepeatedCompositeFieldContainer[DataRow]
    def __init__(self, model_id: _Optional[str] = ..., data: _Optional[_Iterable[_Union[DataRow, _Mapping]]] = ...) -> None: ...

class DataRow(_message.Message):
    __slots__ = ("features",)
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    features: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, features: _Optional[_Iterable[float]] = ...) -> None: ...

class PredictResponse(_message.Message):
    __slots__ = ("status", "model_id", "model_type", "predictions", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: str
    model_id: str
    model_type: str
    predictions: _containers.RepeatedScalarFieldContainer[int]
    error: str
    def __init__(self, status: _Optional[str] = ..., model_id: _Optional[str] = ..., model_type: _Optional[str] = ..., predictions: _Optional[_Iterable[int]] = ..., error: _Optional[str] = ...) -> None: ...

class RetrainModelRequest(_message.Message):
    __slots__ = ("model_id", "hyperparameters")
    class HyperparametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    HYPERPARAMETERS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    hyperparameters: _containers.ScalarMap[str, str]
    def __init__(self, model_id: _Optional[str] = ..., hyperparameters: _Optional[_Mapping[str, str]] = ...) -> None: ...

class RetrainModelResponse(_message.Message):
    __slots__ = ("status", "model_id", "model_type", "hyperparameters", "train_accuracy", "error")
    class HyperparametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    HYPERPARAMETERS_FIELD_NUMBER: _ClassVar[int]
    TRAIN_ACCURACY_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: str
    model_id: str
    model_type: str
    hyperparameters: _containers.ScalarMap[str, str]
    train_accuracy: float
    error: str
    def __init__(self, status: _Optional[str] = ..., model_id: _Optional[str] = ..., model_type: _Optional[str] = ..., hyperparameters: _Optional[_Mapping[str, str]] = ..., train_accuracy: _Optional[float] = ..., error: _Optional[str] = ...) -> None: ...

class DeleteModelRequest(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class DeleteModelResponse(_message.Message):
    __slots__ = ("status", "message", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: str
    message: str
    error: str
    def __init__(self, status: _Optional[str] = ..., message: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class ListDatasetsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListDatasetsResponse(_message.Message):
    __slots__ = ("status", "datasets", "count", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DATASETS_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: str
    datasets: _containers.RepeatedScalarFieldContainer[str]
    count: int
    error: str
    def __init__(self, status: _Optional[str] = ..., datasets: _Optional[_Iterable[str]] = ..., count: _Optional[int] = ..., error: _Optional[str] = ...) -> None: ...
