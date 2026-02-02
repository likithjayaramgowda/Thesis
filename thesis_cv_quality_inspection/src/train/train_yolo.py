import json
from pathlib import Path
from ultralytics import YOLO

from src.utils.config import load_config
from src.utils.seed import set_seed


def make_data_yaml(processed_dir: Path, class_names):
    data_yaml = processed_dir / "yolo_data.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {processed_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"names: {class_names}",
        ]),
        encoding="utf-8"
    )
    return data_yaml


def map_model(name: str) -> str:
    if name == "yolo_nano":
        return "yolov8n.pt"
    if name == "yolo_small":
        return "yolov8s.pt"
    raise ValueError(f"Unknown YOLO model {name}")


def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    out_metrics = Path("outputs/metrics")
    out_metrics.mkdir(parents=True, exist_ok=True)

    for name, dcfg in cfg["data"]["datasets"].items():
        if not dcfg.get("enabled_detection", False):
            continue
        processed_dir = Path(dcfg["processed_dir"])
        labels_dir = processed_dir / "labels"
        if not labels_dir.exists():
            raise RuntimeError(f"YOLO labels folder missing: {labels_dir}")
        label_files = list(labels_dir.rglob("*.txt"))
        if not label_files:
            raise RuntimeError(f"YOLO labels missing in {labels_dir}. Run preprocessing again.")
        # Ensure at least one non-empty label file
        has_labels = False
        for lf in label_files:
            if lf.stat().st_size > 0:
                has_labels = True
                break
        if not has_labels:
            raise RuntimeError(f"YOLO label files are empty in {labels_dir}. Check mask->bbox conversion.")
        data_yaml = make_data_yaml(processed_dir, ["defect"])
        for model_name in cfg["detector"]["models"]:
            model = YOLO(map_model(model_name))
            results = model.train(
                data=str(data_yaml),
                epochs=cfg["detector"]["epochs"],
                imgsz=cfg["detector"]["input_size"],
                batch=cfg["detector"]["batch_size"],
                optimizer="Adam",
                lr0=cfg["detector"]["lr"],
                device="cpu",
                project="outputs/logs",
                name=f"{name}__{model_name}"
            )
            metrics = {
                "dataset": name,
                "model": model_name,
                "train_results_dir": str(results.save_dir),
            }
            out_metrics.joinpath(f"yolo__{name}__{model_name}.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
