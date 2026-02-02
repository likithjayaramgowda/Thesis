import yaml
from pathlib import Path


def load_config(path: str = "configs/experiment_default.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
