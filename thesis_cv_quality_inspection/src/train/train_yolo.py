import json
from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt
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
        labels_dir = processed_dir / "labels_detection"
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
        data_yaml = processed_dir / "yolo_data.yaml"
        if not data_yaml.exists():
            det_names = dcfg["class_names"] if name == "neu" else ["defect"]
            data_yaml = make_data_yaml(processed_dir, det_names)
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

            run_dir = Path(results.save_dir)
            fig_dir = Path("docs/figures")
            fig_dir.mkdir(parents=True, exist_ok=True)
            for fig_name in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png"]:
                src = run_dir / fig_name
                if src.exists():
                    dst = fig_dir / f"FIG_YOLO_{fig_name.replace('.png','')}_{name}__{model_name}.png"
                    shutil.copy2(src, dst)

            results_csv = run_dir / "results.csv"
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                if "epoch" in df.columns:
                    plt.figure(figsize=(8, 4))
                    if "train/box_loss" in df.columns:
                        plt.plot(df["epoch"], df["train/box_loss"], label="train_box_loss")
                    if "val/box_loss" in df.columns:
                        plt.plot(df["epoch"], df["val/box_loss"], label="val_box_loss")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title(f"YOLO Box Loss - {name} - {model_name}")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(fig_dir / f"FIG_YOLO_Loss_{name}__{model_name}.png", dpi=300)
                    plt.close()

                    if "metrics/mAP50(B)" in df.columns:
                        plt.figure(figsize=(8, 4))
                        plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
                        if "metrics/mAP50-95(B)" in df.columns:
                            plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
                        plt.xlabel("Epoch")
                        plt.ylabel("mAP")
                        plt.title(f"YOLO mAP - {name} - {model_name}")
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(fig_dir / f"FIG_YOLO_mAPCurve_{name}__{model_name}.png", dpi=300)
                        plt.close()

            metrics = {
                "dataset": name,
                "model": model_name,
                "train_results_dir": str(results.save_dir),
                "results_csv": str(results_csv) if results_csv.exists() else "",
            }
            out_metrics.joinpath(f"yolo__{name}__{model_name}.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
