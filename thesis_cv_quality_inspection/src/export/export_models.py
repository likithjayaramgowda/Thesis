import json
import shutil
import subprocess
import time
from pathlib import Path

import torch
from torch import nn
from torchvision import models
from ultralytics import YOLO
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


def _append_warning(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    else:
        existing = []
    existing.append(payload)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _load_pruning_acceptance():
    path = Path("outputs/logs/pruning_log.json")
    cls_accept = {}
    det_accept = {}
    if not path.exists():
        return cls_accept, det_accept
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return cls_accept, det_accept

    for item in payload.get("classification_pruned", []):
        ds = item.get("dataset")
        model = item.get("model")
        if ds and model:
            cls_accept[(ds, model)] = bool(item.get("accepted", False))

    for item in payload.get("detection_pruned", []):
        ds = item.get("dataset")
        model = item.get("model")
        if ds and model:
            det_accept[(ds, model)] = bool(item.get("accepted", True))
    return cls_accept, det_accept


def _safe_unlink(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _copy_latest_artifact(ext: str, target_path: Path, run_dir: Path, start_ts: float) -> bool:
    candidates = []
    roots = [Path("runs/export"), run_dir / "weights", run_dir]
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob(f"*{ext}"):
            try:
                if p.stat().st_mtime >= (start_ts - 2.0):
                    candidates.append(p)
            except Exception:
                continue
    if not candidates:
        for root in roots:
            if root.exists():
                candidates.extend(root.rglob(f"*{ext}"))
    if not candidates:
        return False

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, target_path)
    return True


def _export_tflite_from_onnx(onnx_path: Path, tflite_path: Path, extra_args):
    onnx2tf_bin = shutil.which("onnx2tf")
    if not onnx2tf_bin:
        raise RuntimeError("onnx2tf not found on PATH; install onnx2tf to enable classification TFLite export.")

    work_dir = Path("outputs/tmp_onnx2tf") / onnx_path.stem
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [onnx2tf_bin, "-i", str(onnx_path), "-o", str(work_dir), "--non_verbose"]
    cmd.extend(extra_args or [])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"onnx2tf failed ({proc.returncode}): {proc.stderr[-4000:]}")

    tflite_candidates = list(work_dir.rglob("*.tflite"))
    if not tflite_candidates:
        raise RuntimeError("onnx2tf succeeded but no .tflite output was found.")

    def _rank(path: Path):
        name = path.name.lower()
        return (0 if "float32" in name else 1, path.stat().st_mtime)

    selected = sorted(tflite_candidates, key=_rank)[0]
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, tflite_path)


def export_classification(cfg):
    outputs_onnx = Path("outputs/models_onnx")
    outputs_tflite = Path("outputs/models_tflite")
    outputs_onnx.mkdir(parents=True, exist_ok=True)
    outputs_tflite.mkdir(parents=True, exist_ok=True)

    export_cfg = cfg.get("export", {})
    make_tflite = bool(export_cfg.get("classification_tflite_from_onnx", False))
    onnx2tf_extra_args = export_cfg.get("onnx2tf_extra_args", [])
    cls_accept, _ = _load_pruning_acceptance()

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
                onnx_path = outputs_onnx / f"{processed_dir.name}__{model_name}{suffix}.onnx"
                int8_path = outputs_onnx / f"{processed_dir.name}__{model_name}{suffix}_int8.onnx"
                tflite_path = outputs_tflite / f"{processed_dir.name}__{model_name}{suffix}.tflite"

                is_pruned = suffix == "_pruned"
                pruned_ok = cls_accept.get((name, model_name), False) if is_pruned else True
                if not ckpt.exists() or not pruned_ok:
                    _safe_unlink(onnx_path)
                    _safe_unlink(int8_path)
                    _safe_unlink(tflite_path)
                    continue

                model = build_model(model_name, num_classes)
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                model.eval()
                dummy = torch.randn(1, 3, cfg["classification"]["input_size"], cfg["classification"]["input_size"])

                try:
                    torch.onnx.export(
                        model,
                        dummy,
                        onnx_path,
                        opset_version=max(13, cfg["export"]["onnx_opset"]),
                        dynamo=False,
                    )
                except Exception as e:
                    _append_warning(
                        Path("outputs/logs/onnx_export_warnings.json"),
                        {
                            "classification_onnx_export_failed": str(e),
                            "dataset": name,
                            "model": model_name,
                            "variant": suffix,
                        },
                    )
                    continue

                try:
                    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
                except Exception as e:
                    _append_warning(
                        Path("outputs/logs/quantization_warnings.json"),
                        {
                            "classification_quantization_failed": str(e),
                            "dataset": name,
                            "model": model_name,
                            "variant": suffix,
                        },
                    )

                if make_tflite:
                    try:
                        _export_tflite_from_onnx(onnx_path, tflite_path, onnx2tf_extra_args)
                    except Exception as e:
                        _append_warning(
                            Path("outputs/logs/tflite_export_warnings.json"),
                            {
                                "classification_tflite_export_failed": str(e),
                                "dataset": name,
                                "model": model_name,
                                "variant": suffix,
                            },
                        )


def export_yolo(cfg):
    outputs_onnx = Path("outputs/models_onnx")
    outputs_tflite = Path("outputs/models_tflite")
    outputs_onnx.mkdir(parents=True, exist_ok=True)
    outputs_tflite.mkdir(parents=True, exist_ok=True)
    _, det_accept = _load_pruning_acceptance()

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
            variants = [("best.pt", ""), ("best_pruned.pt", "_pruned")]
            for weight_name, suffix in variants:
                pt_path = run_dir / "weights" / weight_name
                onnx_path = outputs_onnx / f"{name}__{model_name}_best{suffix}.onnx"
                onnx_int8_path = outputs_onnx / f"{name}__{model_name}_best{suffix}_int8.onnx"
                tflite_path = outputs_tflite / f"{name}__{model_name}_best{suffix}.tflite"

                is_pruned = suffix == "_pruned"
                pruned_ok = det_accept.get((name, model_name), True) if is_pruned else True
                if not pt_path.exists() or not pruned_ok:
                    _safe_unlink(onnx_path)
                    _safe_unlink(onnx_int8_path)
                    _safe_unlink(tflite_path)
                    continue

                model = YOLO(str(pt_path))
                onnx_start = time.time()
                try:
                    model.export(
                        format="onnx",
                        imgsz=cfg["detector"]["input_size"],
                        opset=cfg["export"]["onnx_opset"],
                        half=False,
                    )
                    copied = _copy_latest_artifact(".onnx", onnx_path, run_dir, onnx_start)
                    if not copied:
                        raise RuntimeError("ONNX export returned but no ONNX artifact could be located.")
                except Exception as e:
                    _append_warning(
                        Path("outputs/logs/export_warnings.json"),
                        {"yolo_onnx_export_failed": str(e), "dataset": name, "model": model_name, "variant": suffix},
                    )
                else:
                    try:
                        quantize_dynamic(str(onnx_path), str(onnx_int8_path), weight_type=QuantType.QInt8)
                    except Exception as e:
                        _append_warning(
                            Path("outputs/logs/quantization_warnings.json"),
                            {
                                "detection_quantization_failed": str(e),
                                "dataset": name,
                                "model": model_name,
                                "variant": suffix,
                            },
                        )

                if cfg["export"].get("tflite_try", False):
                    tflite_start = time.time()
                    try:
                        model.export(format="tflite", imgsz=cfg["detector"]["input_size"])
                        copied = _copy_latest_artifact(".tflite", tflite_path, run_dir, tflite_start)
                        if not copied:
                            raise RuntimeError("TFLite export returned but no TFLite artifact could be located.")
                    except Exception as e:
                        _append_warning(
                            Path("outputs/logs/export_warnings.json"),
                            {"yolo_tflite_export_failed": str(e), "dataset": name, "model": model_name, "variant": suffix},
                        )


def main():
    cfg = load_config()
    export_classification(cfg)
    export_yolo(cfg)


if __name__ == "__main__":
    main()
