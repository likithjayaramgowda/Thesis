import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from PIL import Image
import onnxruntime as ort
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ultralytics import YOLO

from src.utils.config import load_config
from src.utils.energy import sample_cpu_util
from src.utils.threads import set_thread_count


def _clear_previous_metrics(metrics_out: Path):
    for pattern in [
        "classification_bench__*.json",
        "detection_bench__*.json",
    ]:
        for f in metrics_out.glob(pattern):
            try:
                f.unlink()
            except Exception:
                pass


def _load_pruning_acceptance():
    path = Path("outputs/logs/pruning_log.json")
    accepted = {}
    if not path.exists():
        return accepted
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return accepted
    for item in payload.get("classification_pruned", []):
        dataset = item.get("dataset")
        model = item.get("model")
        if dataset and model:
            accepted[(dataset, model)] = bool(item.get("accepted", False))
    for item in payload.get("detection_pruned", []):
        dataset = item.get("dataset")
        model = item.get("model")
        if dataset and model:
            accepted[(dataset, model)] = bool(item.get("accepted", True))
    return accepted


def load_image(path: Path, size: int):
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...]


def _resolve_class_names(processed_dir: Path, fallback_names):
    train_root = processed_dir / "images" / "train"
    if train_root.exists():
        names = sorted([p.name for p in train_root.iterdir() if p.is_dir()])
        if names:
            return names, "train_imagefolder_sorted"
    return list(fallback_names), "config_class_names"


def _build_cls_model(name: str, num_classes: int):
    from torch import nn
    from torchvision import models

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


def _predict_pytorch(ckpt_path: Path, model_name: str, num_classes: int, images):
    model = _build_cls_model(model_name, num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    preds = []
    with torch.no_grad():
        for img in images:
            x = torch.from_numpy(img).float()
            out = model(x)
            preds.append(int(out.argmax(1).item()))
    return preds


def _predict_onnx(model_path: Path, images):
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    preds = []
    for img in images:
        out = sess.run(None, {input_name: img})[0]
        preds.append(int(np.argmax(out)))
    return preds


def _set_tflite_input(interpreter, input_detail, img_nchw):
    img_nhwc = np.transpose(img_nchw, (0, 2, 3, 1))
    dtype = input_detail["dtype"]
    if dtype in (np.uint8, np.int8):
        scale, zero_point = input_detail.get("quantization", (0.0, 0))
        if scale and scale > 0:
            q = np.round(img_nhwc / scale + zero_point)
        else:
            q = img_nhwc
        info = np.iinfo(dtype)
        arr = np.clip(q, info.min, info.max).astype(dtype)
    else:
        arr = img_nhwc.astype(dtype)
    interpreter.set_tensor(input_detail["index"], arr)


def _get_tflite_output(interpreter, output_detail):
    out = interpreter.get_tensor(output_detail["index"])
    dtype = output_detail["dtype"]
    if dtype in (np.uint8, np.int8):
        scale, zero_point = output_detail.get("quantization", (0.0, 0))
        if scale and scale > 0:
            out = (out.astype(np.float32) - zero_point) * scale
        else:
            out = out.astype(np.float32)
    return out


def _predict_tflite(model_path: Path, images):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    preds = []
    for img in images:
        _set_tflite_input(interpreter, input_detail, img)
        interpreter.invoke()
        out = _get_tflite_output(interpreter, output_detail)
        preds.append(int(np.argmax(out)))
    return preds


def benchmark_onnx(model_path: Path, images, threads: int):
    so = ort.SessionOptions()
    if threads > 0:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = threads
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    if not images:
        return 0.0, 0.0
    latencies = []
    for img in images:
        start = time.time()
        sess.run(None, {input_name: img})
        latencies.append((time.time() - start) * 1000.0)
    avg_latency = float(sum(latencies) / len(latencies))
    fps = float(1000.0 / avg_latency) if avg_latency > 0 else 0.0
    return avg_latency, fps


def benchmark_tflite(model_path: Path, images):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if not images:
        return 0.0, 0.0
    latencies = []
    for img in images:
        start = time.time()
        _set_tflite_input(interpreter, input_detail, img)
        interpreter.invoke()
        _ = _get_tflite_output(interpreter, output_detail)
        latencies.append((time.time() - start) * 1000.0)
    avg_latency = float(sum(latencies) / len(latencies))
    fps = float(1000.0 / avg_latency) if avg_latency > 0 else 0.0
    return avg_latency, fps


def _classification_runtime_specs(dataset: str, model_name: str, suffix: str):
    specs = []
    onnx_fp32 = Path("outputs/models_onnx") / f"{dataset}__{model_name}{suffix}.onnx"
    onnx_int8 = Path("outputs/models_onnx") / f"{dataset}__{model_name}{suffix}_int8.onnx"
    tflite = Path("outputs/models_tflite") / f"{dataset}__{model_name}{suffix}.tflite"

    if onnx_fp32.exists():
        specs.append({"runtime": "onnx", "onnx_variant": "fp32", "path": onnx_fp32})
    if onnx_int8.exists():
        specs.append({"runtime": "onnx", "onnx_variant": "int8", "path": onnx_int8})
    if tflite.exists():
        specs.append({"runtime": "tflite", "onnx_variant": "", "path": tflite})
    return specs


def bench_classification(cfg):
    metrics_out = Path("outputs/metrics")
    metrics_out.mkdir(parents=True, exist_ok=True)
    root = Path(".").resolve()
    profiles = cfg["edge_profiles"]
    energy_tdp = cfg["energy_proxy"]["tdp_watts"]
    input_size = int(cfg["classification"]["input_size"])
    bench_cfg = cfg.get("benchmark", {})
    max_samples = int(bench_cfg.get("max_samples", 240))
    latency_samples = int(bench_cfg.get("latency_samples", 20))
    prune_accept = _load_pruning_acceptance()

    for name, dcfg in cfg["data"]["datasets"].items():
        processed_dir = Path(dcfg["processed_dir"])
        if not processed_dir.exists():
            continue
        labels_path = processed_dir / "labels_classification.csv"
        splits_path = processed_dir / "splits.csv"
        if not labels_path.exists() or not splits_path.exists():
            continue

        labels = pd.read_csv(labels_path)
        splits = pd.read_csv(splits_path)
        test = labels.merge(splits, on="image_path")
        test = test[test["split"] == "test"]
        if max_samples > 0:
            test = test.head(max_samples)
        if test.empty:
            continue

        class_names, class_index_source = _resolve_class_names(processed_dir, dcfg["class_names"])
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        images, y_true = [], []
        for _, row in test.iterrows():
            label = row["class"]
            if label not in class_to_idx:
                continue
            img_path = Path(row["image_path"])
            if not img_path.is_absolute():
                img_path = root / img_path
            if not img_path.exists():
                continue
            images.append(load_image(img_path, input_size))
            y_true.append(class_to_idx[label])
        if not images:
            continue

        for model_name in cfg["classification"]["models"]:
            variants = [("", model_name), ("_pruned", f"{model_name}_pruned")]
            for suffix, label in variants:
                if suffix == "_pruned" and not prune_accept.get((name, model_name), False):
                    continue
                runtime_specs = _classification_runtime_specs(name, model_name, suffix)
                if not runtime_specs:
                    continue

                ckpt_path = Path("outputs/checkpoints") / f"{name}__{model_name}{suffix if suffix else ''}_best.pth"
                if suffix == "_pruned":
                    ckpt_path = Path("outputs/checkpoints") / f"{name}__{model_name}_pruned.pth"
                ref_pt_acc = None
                try:
                    if ckpt_path.exists():
                        pt_preds = _predict_pytorch(ckpt_path, model_name, len(class_names), images)
                        ref_pt_acc = float(accuracy_score(y_true, pt_preds))
                except Exception:
                    ref_pt_acc = None

                for runtime_spec in runtime_specs:
                    runtime = runtime_spec["runtime"]
                    model_path = runtime_spec["path"]
                    onnx_variant = runtime_spec["onnx_variant"]
                    try:
                        if runtime == "onnx":
                            y_pred = _predict_onnx(model_path, images)
                        else:
                            y_pred = _predict_tflite(model_path, images)
                    except Exception:
                        continue

                    acc = float(accuracy_score(y_true, y_pred))
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average="macro", zero_division=0
                    )
                    acc_gap = float(acc - ref_pt_acc) if ref_pt_acc is not None else None

                    for profile, threads in profiles.items():
                        set_thread_count(threads)
                        avg_cpu = sample_cpu_util()
                        try:
                            if runtime == "onnx":
                                latency, fps = benchmark_onnx(model_path, images[:latency_samples], threads)
                            else:
                                latency, fps = benchmark_tflite(model_path, images[:latency_samples])
                        except Exception:
                            continue
                        energy_wh = (avg_cpu / 100.0) * energy_tdp * (latency / 1000.0) / 3600.0 * 1000.0
                        record = {
                            "dataset": name,
                            "model": label,
                            "profile": profile,
                            "runtime": runtime,
                            "accuracy": acc,
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1": float(f1),
                            "latency_ms": latency,
                            "fps": fps,
                            "model_size_mb": model_path.stat().st_size / (1024 * 1024),
                            "energy_proxy_wh_per_1000": energy_wh,
                            "energy_method": cfg["energy_proxy"]["method"],
                            "onnx_variant": onnx_variant,
                            "reference_pt_accuracy": ref_pt_acc,
                            "accuracy_gap_vs_pt": acc_gap,
                            "class_index_source": class_index_source,
                        }
                        suffix_name = f"__{runtime}{'_' + onnx_variant if onnx_variant else ''}"
                        metrics_out.joinpath(
                            f"classification_bench__{name}__{label}__{profile}{suffix_name}.json"
                        ).write_text(json.dumps(record, indent=2), encoding="utf-8")


def _detection_runtime_specs(dataset: str, model_name: str, suffix: str, onnx_path: Path):
    specs = []
    if onnx_path.exists():
        specs.append({"runtime": "onnx", "path": onnx_path, "onnx_variant": "fp32"})
    onnx_int8 = Path("outputs/models_onnx") / f"{dataset}__{model_name}_best{suffix}_int8.onnx"
    if onnx_int8.exists():
        specs.append({"runtime": "onnx", "path": onnx_int8, "onnx_variant": "int8"})
    tflite_path = Path("outputs/models_tflite") / f"{dataset}__{model_name}_best{suffix}.tflite"
    if tflite_path.exists():
        specs.append({"runtime": "tflite", "path": tflite_path, "onnx_variant": ""})
    return specs


def _collect_detection_images(processed_dir: Path):
    test_dir = processed_dir / "yolo" / "images" / "test"
    if not test_dir.exists():
        test_dir = processed_dir / "images_detection" / "test"
    sample_imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        sample_imgs.extend(test_dir.rglob(ext))
    return sample_imgs


def bench_detection(cfg):
    metrics_out = Path("outputs/metrics")
    metrics_out.mkdir(parents=True, exist_ok=True)
    profiles = cfg["edge_profiles"]
    input_size = int(cfg["detector"]["input_size"])
    energy_tdp = cfg["energy_proxy"]["tdp_watts"]
    target_fps = float(cfg.get("realtime_sim", {}).get("target_fps", 30))
    bench_cfg = cfg.get("benchmark", {})
    latency_samples = int(bench_cfg.get("latency_samples", 20))
    prune_accept = _load_pruning_acceptance()

    for name, dcfg in cfg["data"]["datasets"].items():
        if not dcfg.get("enabled_detection", False):
            continue
        processed_dir = Path(dcfg["processed_dir"])
        if not processed_dir.exists():
            continue

        data_yaml = processed_dir / "yolo_data.yaml"
        if not data_yaml.exists():
            continue

        def find_run_dir(dataset, model):
            candidates = [
                Path("outputs/logs") / f"{dataset}__{model}",
                Path("runs/detect/outputs/logs") / f"{dataset}__{model}",
            ]
            for c in candidates:
                if c.exists():
                    return c
            return None

        images = [load_image(p, input_size) for p in _collect_detection_images(processed_dir)[:latency_samples]]

        for model_name in cfg["detector"]["models"]:
            run_dir = find_run_dir(name, model_name)
            if not run_dir:
                continue
            variants = [("best.pt", model_name, ""), ("best_pruned.pt", f"{model_name}_pruned", "_pruned")]
            for pt_name, label, suffix in variants:
                if suffix == "_pruned" and not prune_accept.get((name, model_name), False):
                    continue
                pt_path = run_dir / "weights" / pt_name
                if not pt_path.exists():
                    continue

                onnx_path = Path("outputs/models_onnx") / f"{name}__{model_name}_best{suffix}.onnx"
                if not onnx_path.exists():
                    alt = run_dir / "weights" / "best.onnx"
                    if alt.exists():
                        onnx_path = alt

                model = YOLO(str(pt_path))
                val_res = model.val(data=str(data_yaml), imgsz=input_size, device="cpu")
                map50 = float(val_res.box.map50) if hasattr(val_res.box, "map50") else 0.0
                map5095 = float(val_res.box.map) if hasattr(val_res.box, "map") else 0.0

                def _scalar(v):
                    try:
                        if isinstance(v, np.ndarray):
                            return float(np.mean(v)) if v.size else 0.0
                    except Exception:
                        pass
                    try:
                        return float(v)
                    except Exception:
                        return 0.0

                runtime_specs = _detection_runtime_specs(name, model_name, suffix, onnx_path)
                for runtime_spec in runtime_specs:
                    runtime = runtime_spec["runtime"]
                    model_path = runtime_spec["path"]
                    onnx_variant = runtime_spec["onnx_variant"]
                    for profile, threads in profiles.items():
                        set_thread_count(threads)
                        avg_cpu = sample_cpu_util()
                        try:
                            if runtime == "onnx":
                                latency, fps = benchmark_onnx(model_path, images, threads)
                            else:
                                latency, fps = benchmark_tflite(model_path, images)
                        except Exception:
                            continue
                        energy_wh = (avg_cpu / 100.0) * energy_tdp * (latency / 1000.0) / 3600.0 * 1000.0

                        record = {
                            "dataset": name,
                            "model": label,
                            "profile": profile,
                            "runtime": runtime,
                            "onnx_variant": onnx_variant,
                            "map50": map50,
                            "map5095": map5095,
                            "precision": _scalar(val_res.box.p) if hasattr(val_res.box, "p") else 0.0,
                            "recall": _scalar(val_res.box.r) if hasattr(val_res.box, "r") else 0.0,
                            "latency_ms": latency,
                            "fps": fps,
                            "realtime_target_fps": target_fps,
                            "realtime_target_met": bool(fps >= target_fps),
                            "model_size_mb": model_path.stat().st_size / (1024 * 1024),
                            "energy_proxy_wh_per_1000": energy_wh,
                            "energy_method": cfg["energy_proxy"]["method"],
                            "note": f"mAP computed with PT weights; latency from {runtime.upper()}",
                        }
                        suffix_name = f"__{runtime}{'_' + onnx_variant if onnx_variant else ''}"
                        metrics_out.joinpath(
                            f"detection_bench__{name}__{label}__{profile}{suffix_name}.json"
                        ).write_text(json.dumps(record, indent=2), encoding="utf-8")


def main():
    cfg = load_config()
    metrics_out = Path("outputs/metrics")
    metrics_out.mkdir(parents=True, exist_ok=True)
    _clear_previous_metrics(metrics_out)
    bench_classification(cfg)
    bench_detection(cfg)


if __name__ == "__main__":
    main()
