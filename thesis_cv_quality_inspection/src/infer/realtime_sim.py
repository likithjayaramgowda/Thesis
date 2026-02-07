import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort

from src.utils.config import load_config
from src.utils.threads import set_thread_count
from src.utils.energy import sample_cpu_util


def _load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr


def _softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def _pick_images(processed_dir: Path, max_frames: int):
    labels = pd.read_csv(processed_dir / "labels_classification.csv")
    splits = pd.read_csv(processed_dir / "splits.csv")
    test = labels.merge(splits, on="image_path")
    test = test[test["split"] == "test"]
    paths = []
    for p in test["image_path"].tolist():
        path = Path(p)
        if not path.is_absolute():
            path = Path(".").resolve() / path
        if path.exists():
            paths.append(path)
        if len(paths) >= max_frames:
            break
    return paths


def run_realtime_sim():
    cfg = load_config()
    input_size = cfg["classification"]["input_size"]
    profiles = cfg["edge_profiles"]
    sim_cfg = cfg.get("realtime_sim", {})
    target_fps = float(sim_cfg.get("target_fps", 30))
    max_frames = int(sim_cfg.get("max_frames", 200))
    sim_profiles = sim_cfg.get("profiles", [])
    sim_mode = sim_cfg.get("mode", "dataset_stream")

    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, dcfg in cfg["data"]["datasets"].items():
        if not dcfg.get("enabled_classification", False):
            continue
        processed_dir = Path(dcfg["processed_dir"])
        if not processed_dir.exists():
            continue
        images = _pick_images(processed_dir, max_frames)
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
                if not onnx_path.exists():
                    continue

                try:
                    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                except Exception:
                    if onnx_path != onnx_fp32 and onnx_fp32.exists():
                        onnx_path = onnx_fp32
                        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                    else:
                        continue
                input_name = sess.get_inputs()[0].name
                imgs = [_load_image(p, input_size) for p in images]

                for profile, threads in profiles.items():
                    if sim_profiles and profile not in sim_profiles:
                        continue
                    set_thread_count(threads)
                    avg_cpu = sample_cpu_util()
                    latencies = []
                    confidences = []
                    for img in imgs:
                        start = time.time()
                        out = sess.run(None, {input_name: img})[0][0]
                        latencies.append((time.time() - start) * 1000.0)
                        conf = float(np.max(_softmax(out)))
                        confidences.append(conf)

                    latencies = np.array(latencies)
                    mean_latency = float(latencies.mean()) if latencies.size else 0.0
                    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0
                    std_fps = float(np.std(1000.0 / np.maximum(latencies, 1e-6)))
                    violations = int(np.sum(latencies > (1000.0 / max(target_fps, 1e-6))))
                    energy_wh = (avg_cpu / 100.0) * cfg["energy_proxy"]["tdp_watts"] * (mean_latency/1000.0) / 3600.0 * 1000.0

                    record = {
                        "dataset": name,
                        "model": label,
                        "profile": profile,
                        "stream_mode": sim_mode,
                        "hardware_mode": "cpu_only_no_extra_hardware",
                        "target_fps": target_fps,
                        "frames": len(imgs),
                        "mean_latency_ms": mean_latency,
                        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies.size else 0.0,
                        "mean_fps": fps,
                        "std_fps": std_fps,
                        "violations": violations,
                        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
                        "energy_proxy_wh_per_1000": energy_wh,
                        "energy_method": cfg["energy_proxy"]["method"],
                        "onnx_variant": "int8" if onnx_path.name.endswith("_int8.onnx") else "fp32",
                    }
                    out_dir.joinpath(f"realtime_sim__{name}__{label}__{profile}.json").write_text(
                        json.dumps(record, indent=2), encoding="utf-8"
                    )


if __name__ == "__main__":
    run_realtime_sim()
