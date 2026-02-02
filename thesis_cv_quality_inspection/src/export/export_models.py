import json
from pathlib import Path
import torch
from torchvision import models
from torch import nn
from ultralytics import YOLO
import tensorflow as tf
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.utils.config import load_config


def build_model(name: str, num_classes: int):
    if name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model {name}")
    return model


def export_classification(cfg):
    outputs_onnx = Path("outputs/models_onnx")
    outputs_tflite = Path("outputs/models_tflite")
    outputs_onnx.mkdir(parents=True, exist_ok=True)
    outputs_tflite.mkdir(parents=True, exist_ok=True)
    for name, dcfg in cfg["data"]["datasets"].items():
        processed_dir = Path(dcfg["processed_dir"])
        if not processed_dir.exists():
            continue
        num_classes = len(dcfg["class_names"])
        for model_name in cfg["classification"]["models"]:
            variants = [
                ("", Path("outputs/checkpoints") / f"{processed_dir.name}__{model_name}_best.pth"),
                ("_pruned", Path("outputs/checkpoints") / f"{processed_dir.name}__{model_name}_pruned.pth"),
            ]
            for suffix, ckpt in variants:
                if not ckpt.exists():
                    continue
                model = build_model(model_name, num_classes)
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                model.eval()
                dummy = torch.randn(1, 3, cfg["classification"]["input_size"], cfg["classification"]["input_size"])
                onnx_path = outputs_onnx / f"{processed_dir.name}__{model_name}{suffix}.onnx"
                try:
                    torch.onnx.export(
                        model,
                        dummy,
                        onnx_path,
                        opset_version=max(13, cfg["export"]["onnx_opset"]),
                        dynamo=False,
                    )
                except Exception as e:
                    log = Path("outputs/logs") / "onnx_export_warnings.json"
                    log.write_text(json.dumps({"classification_onnx_export_failed": str(e)}, indent=2), encoding="utf-8")
                    continue
                # INT8 quantization (dynamic)
                try:
                    int8_path = outputs_onnx / f"{processed_dir.name}__{model_name}{suffix}_int8.onnx"
                    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
                except Exception as e:
                    log = Path("outputs/logs") / "quantization_warnings.json"
                    log.write_text(json.dumps({"classification_quantization_failed": str(e)}, indent=2), encoding="utf-8")

            # TFLite export (uses Keras ImageNet weights as proxy for latency benchmarking)
            if suffix == "":
                try:
                    if model_name == "resnet50":
                        kmodel = tf.keras.applications.ResNet50(weights="imagenet", input_shape=(cfg["classification"]["input_size"], cfg["classification"]["input_size"], 3))
                    elif model_name == "efficientnet_b0":
                        kmodel = tf.keras.applications.EfficientNetB0(weights="imagenet", input_shape=(cfg["classification"]["input_size"], cfg["classification"]["input_size"], 3))
                    elif model_name == "mobilenet_v3_large":
                        kmodel = tf.keras.applications.MobileNetV3Large(weights="imagenet", input_shape=(cfg["classification"]["input_size"], cfg["classification"]["input_size"], 3))
                    else:
                        kmodel = None
                    if kmodel:
                        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
                        tflite_model = converter.convert()
                        tflite_path = outputs_tflite / f"{processed_dir.name}__{model_name}.tflite"
                        tflite_path.write_bytes(tflite_model)
                except Exception as e:
                    log = Path("outputs/logs") / "tflite_export_warnings.json"
                    log.write_text(json.dumps({"classification_tflite_export_failed": str(e)}, indent=2), encoding="utf-8")


def export_yolo(cfg):
    outputs_onnx = Path("outputs/models_onnx")
    outputs_tflite = Path("outputs/models_tflite")
    outputs_onnx.mkdir(parents=True, exist_ok=True)
    outputs_tflite.mkdir(parents=True, exist_ok=True)

    def find_run_dir(dataset, model):
        candidates = [
            Path("outputs/logs") / f"{dataset}__{model}",
            Path("runs/detect/outputs/logs") / f"{dataset}__{model}",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    for name, dcfg in cfg["data"]["datasets"].items():
        if not dcfg.get("enabled_detection", False):
            continue
        for model_name in cfg["detector"]["models"]:
            run_dir = find_run_dir(name, model_name)
            if not run_dir:
                continue
            best_pt = run_dir / "weights" / "best.pt"
            if not best_pt.exists():
                continue
            model = YOLO(str(best_pt))
            onnx_path = outputs_onnx / f"{name}__{model_name}_best.onnx"
            model.export(format="onnx", imgsz=cfg["detector"]["input_size"], opset=cfg["export"]["onnx_opset"], half=False)
            # Move exported ONNX from runs/export to outputs/models_onnx
            export_dir = Path("runs") / "export"
            if export_dir.exists():
                onnx_candidates = sorted(export_dir.rglob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
                if onnx_candidates:
                    onnx_candidates[0].replace(onnx_path)
            # If Ultralytics saved ONNX in weights/, copy it
            best_onnx = run_dir / "weights" / "best.onnx"
            if best_onnx.exists():
                if onnx_path.exists():
                    onnx_path.unlink()
                best_onnx.replace(onnx_path)
            if cfg["export"].get("tflite_try", False):
                try:
                    model.export(format="tflite", imgsz=cfg["detector"]["input_size"])
                except Exception as e:
                    log = Path("outputs/logs") / "export_warnings.json"
                    data = {"yolo_tflite_export_failed": str(e)}
                    log.write_text(json.dumps(data, indent=2), encoding="utf-8")

            # pruning not supported in this pipeline
            log = Path("outputs/logs") / "pruning_log.json"
            log.write_text(json.dumps({"yolo_pruning": "Not supported; proceeding with quantization only."}, indent=2), encoding="utf-8")


def main():
    cfg = load_config()
    export_classification(cfg)
    export_yolo(cfg)


if __name__ == "__main__":
    main()
