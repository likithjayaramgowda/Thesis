import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ultralytics import YOLO

from src.utils.config import load_config
from src.utils.threads import set_thread_count
from src.utils.energy import sample_cpu_util


def load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr


def benchmark_onnx(model_path, images, input_size, threads):
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
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    return avg_latency, fps


def bench_classification(cfg):
    metrics_out = Path("outputs/metrics")
    metrics_out.mkdir(parents=True, exist_ok=True)
    root = Path(".").resolve()
    profiles = cfg["edge_profiles"]
    energy_tdp = cfg["energy_proxy"]["tdp_watts"]
    input_size = cfg["classification"]["input_size"]

    for name, dcfg in cfg["data"]["datasets"].items():
        processed_dir = Path(dcfg["processed_dir"])
        if not processed_dir.exists():
            continue
        labels = pd.read_csv(processed_dir / "labels_classification.csv")
        splits = pd.read_csv(processed_dir / "splits.csv")
        test = labels.merge(splits, on="image_path")
        test = test[test["split"] == "test"]
        if test.empty:
            continue

        y_true = []
        images = []
        for _, row in test.iterrows():
            img_path = Path(row["image_path"])
            if not img_path.is_absolute():
                img_path = root / img_path
            if not img_path.exists():
                continue
            images.append(load_image(img_path, input_size))
            y_true.append(dcfg["class_names"].index(row["class"]))
        if not images:
            continue

        for model_name in cfg["classification"]["models"]:
            variants = [
                ("", model_name),
                ("_pruned", f"{model_name}_pruned"),
            ]
            for suffix, label in variants:
                onnx_int8 = Path("outputs/models_onnx") / f"{name}__{model_name}{suffix}_int8.onnx"
                onnx_fp32 = Path("outputs/models_onnx") / f"{name}__{model_name}{suffix}.onnx"
                onnx_path = onnx_int8 if onnx_int8.exists() else onnx_fp32
                tflite_path = Path("outputs/models_tflite") / f"{name}__{model_name}.tflite"
                if not onnx_path.exists():
                    continue
                try:
                    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                except Exception:
                    # fallback to fp32 if int8 unsupported by ORT build
                    onnx_path = onnx_fp32
                    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                input_name = sess.get_inputs()[0].name
                y_pred = []
                for img in images:
                    out = sess.run(None, {input_name: img})[0]
                    y_pred.append(int(np.argmax(out)))

                acc = accuracy_score(y_true, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

                for profile, threads in profiles.items():
                    set_thread_count(threads)
                    avg_cpu = sample_cpu_util()
                    latency, fps = benchmark_onnx(onnx_path, images[:20], input_size, threads)
                    energy_wh = (avg_cpu / 100.0) * energy_tdp * (latency/1000.0) / 3600.0 * 1000.0
                    record = {
                        "dataset": name,
                        "model": label,
                        "profile": profile,
                        "runtime": "onnx",
                        "accuracy": acc,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "latency_ms": latency,
                        "fps": fps,
                        "model_size_mb": onnx_path.stat().st_size / (1024 * 1024),
                        "energy_proxy_wh_per_1000": energy_wh,
                        "energy_method": cfg["energy_proxy"]["method"],
                        "onnx_variant": "int8" if onnx_path.name.endswith("_int8.onnx") else "fp32",
                    }
                    metrics_out.joinpath(f"classification_bench__{name}__{label}__{profile}.json").write_text(
                        json.dumps(record, indent=2), encoding="utf-8"
                    )

                if suffix == "" and tflite_path.exists():
                    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
                    interpreter.allocate_tensors()
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    for profile, threads in profiles.items():
                        set_thread_count(threads)
                        latencies = []
                        for img in images[:20]:
                            start = time.time()
                            interpreter.set_tensor(input_details[0]["index"], img.transpose(0,2,3,1))
                            interpreter.invoke()
                            _ = interpreter.get_tensor(output_details[0]["index"])
                            latencies.append((time.time() - start) * 1000.0)
                        latency = sum(latencies) / len(latencies)
                        fps = 1000.0 / latency if latency > 0 else 0.0
                        record = {
                            "dataset": name,
                            "model": label,
                            "profile": profile,
                            "runtime": "tflite",
                            "accuracy": acc,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "latency_ms": latency,
                            "fps": fps,
                            "model_size_mb": tflite_path.stat().st_size / (1024 * 1024),
                            "energy_proxy_wh_per_1000": 0.0,
                            "energy_method": "cpu_util_proxy"
                        }
                        metrics_out.joinpath(f"classification_bench__{name}__{label}__{profile}__tflite.json").write_text(
                            json.dumps(record, indent=2), encoding="utf-8"
                        )


def bench_detection(cfg):
    metrics_out = Path("outputs/metrics")
    metrics_out.mkdir(parents=True, exist_ok=True)
    profiles = cfg["edge_profiles"]
    input_size = cfg["detector"]["input_size"]
    energy_tdp = cfg["energy_proxy"]["tdp_watts"]

    for name, dcfg in cfg["data"]["datasets"].items():
        if not dcfg.get("enabled_detection", False):
            continue
        processed_dir = Path(dcfg["processed_dir"])
        if not processed_dir.exists():
            continue
        data_yaml = processed_dir / "yolo_data.yaml"
        if not data_yaml.exists():
            names = dcfg["class_names"] if name == "neu" else ["defect"]
            names_serialized = "[" + ", ".join([f"'{c}'" for c in names]) + "]"
            data_yaml.write_text(
                "\n".join(
                    [
                        f"path: {processed_dir.resolve().as_posix()}",
                        "train: images_detection/train",
                        "val: images_detection/val",
                        "test: images_detection/test",
                        f"names: {names_serialized}",
                    ]
                ),
                encoding="utf-8",
            )
        def find_run_dir(dataset, model):
            candidates = [
                Path("outputs/logs") / f"{dataset}__{model}",
                Path("runs/detect/outputs/logs") / f"{dataset}__{model}",
            ]
            for c in candidates:
                if c.exists():
                    return c
            return None

        for model_name in cfg["detector"]["models"]:
            run_dir = find_run_dir(name, model_name)
            if not run_dir:
                continue
            variants = [("best.pt", model_name, ""), ("best_pruned.pt", f"{model_name}_pruned", "_pruned")]
            for pt_name, label, onnx_suffix in variants:
                pt_path = run_dir / "weights" / pt_name
                if not pt_path.exists():
                    continue
                onnx_path = Path("outputs/models_onnx") / f"{name}__{model_name}_best{onnx_suffix}.onnx"
                if not onnx_path.exists():
                    alt = run_dir / "weights" / "best.onnx"
                    if alt.exists():
                        onnx_path = alt

                model = YOLO(str(pt_path))
                val_res = model.val(data=str(data_yaml), imgsz=input_size, device="cpu")
                map50 = float(val_res.box.map50) if hasattr(val_res.box, "map50") else 0.0
                map5095 = float(val_res.box.map) if hasattr(val_res.box, "map") else 0.0

                if onnx_path.exists():
                    sample_imgs = list((processed_dir / "images_detection" / "test").rglob("*.jpg"))
                    if not sample_imgs:
                        sample_imgs = list((processed_dir / "images_detection" / "test").rglob("*.bmp"))
                    if not sample_imgs:
                        sample_imgs = list((processed_dir / "images_detection" / "test").rglob("*.png"))
                    sample_imgs = sample_imgs[:10]
                    images = [load_image(p, input_size) for p in sample_imgs]
                    for profile, threads in profiles.items():
                        set_thread_count(threads)
                        avg_cpu = sample_cpu_util()
                        latency, fps = benchmark_onnx(onnx_path, images, input_size, threads)
                        energy_wh = (avg_cpu / 100.0) * energy_tdp * (latency/1000.0) / 3600.0 * 1000.0

                        def _scalar(v):
                            try:
                                import numpy as _np
                                if isinstance(v, _np.ndarray):
                                    return float(_np.mean(v)) if v.size else 0.0
                            except Exception:
                                pass
                            try:
                                return float(v)
                            except Exception:
                                return 0.0

                        record = {
                            "dataset": name,
                            "model": label,
                            "profile": profile,
                            "map50": map50,
                            "map5095": map5095,
                            "precision": _scalar(val_res.box.p) if hasattr(val_res.box, "p") else 0.0,
                            "recall": _scalar(val_res.box.r) if hasattr(val_res.box, "r") else 0.0,
                            "latency_ms": latency,
                            "fps": fps,
                            "model_size_mb": onnx_path.stat().st_size / (1024 * 1024) if onnx_path.exists() else 0.0,
                            "energy_proxy_wh_per_1000": energy_wh,
                            "energy_method": cfg["energy_proxy"]["method"],
                            "note": "mAP computed with PT weights; latency from ONNX",
                        }
                        metrics_out.joinpath(f"detection_bench__{name}__{label}__{profile}.json").write_text(
                            json.dumps(record, indent=2), encoding="utf-8"
                        )


def main():
    cfg = load_config()
    bench_classification(cfg)
    bench_detection(cfg)


if __name__ == "__main__":
    main()
