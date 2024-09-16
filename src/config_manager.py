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

