from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    source: str
    sourceURL: Path
    username: str
    extract_to: Path


@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    model_type: str
    input_img_size: list
    params_lr: float
    include_top: bool
    weights: str
    classes: int
    optimizer: str
    loss_function: str


@dataclass(frozen=True)
class DataPreprocessingConfig:
    training_data: Path
    batch_size: int
    is_augmentation: bool
    img_size: list


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    trained_model_path: Path
    base_model_path: Path
    training_data: Path
    n_epochs: int
    batch_size: int
    is_augmentation: bool
    img_size: list


@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_path: Path
    training_data: Path
    all_params: dict


@dataclass(frozen=True)
class MLFlowConfig:
    mlflow_uri: str


