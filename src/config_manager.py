from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    source: str
    sourceURL: Path
    username: str
    extract_to: Path


