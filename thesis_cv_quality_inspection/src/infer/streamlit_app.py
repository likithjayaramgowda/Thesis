import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from src.utils.config import load_config
from src.utils.threads import set_thread_count


def _softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def _preprocess_classification(frame: np.ndarray, input_size: int):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size))
    arr = img.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr


def _iter_dataset_images(dataset: str, task: str, max_frames: int):
    if task == "classification":
        root = Path(f"data/processed/{dataset}/images/test")
    else:
        root = Path(f"data/processed/{dataset}/yolo/images/test")
        if not root.exists():
            root = Path(f"data/processed/{dataset}/images_detection/test")
    imgs = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        imgs.extend(list(root.rglob(ext)))
    imgs = sorted(imgs)[:max_frames]
    for p in imgs:
        frame = cv2.imread(str(p))
        if frame is not None:
            yield frame, p.name


def _iter_video_frames(video_bytes: bytes, max_frames: int):
    tmp = Path("outputs") / "tmp_streamlit_video.mp4"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(video_bytes)
    cap = cv2.VideoCapture(str(tmp))
    count = 0
    while cap.isOpened() and count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
        yield frame, f"frame_{count:04d}"
    cap.release()


def main():
    st.set_page_config(page_title="CV QC Realtime Simulation", layout="wide")
    st.title("Real-Time Video Simulation (CPU Only)")

    cfg = load_config()
    profiles = cfg["edge_profiles"]
    dataset_options = [k for k, v in cfg["data"]["datasets"].items() if Path(v["processed_dir"]).exists()]

    with st.sidebar:
        task = st.selectbox("Task", ["classification", "detection"])
        dataset = st.selectbox("Dataset", dataset_options, index=0 if dataset_options else None)
        profile = st.selectbox("CPU Profile", list(profiles.keys()), index=0)
        target_fps = st.slider("Target FPS", 1, 60, int(cfg.get("realtime_sim", {}).get("target_fps", 30)))
        max_frames = st.slider("Max Frames", 5, 500, 100)
        source = st.radio("Source", ["Dataset Stream", "Uploaded Video"])
        video_file = st.file_uploader("Upload video (.mp4/.avi)", type=["mp4", "avi", "mov"]) if source == "Uploaded Video" else None
        start = st.button("Start Simulation")

    if not dataset:
        st.warning("No processed datasets found. Run preprocessing first.")
        return

    threads = profiles.get(profile, 1)
    set_thread_count(threads)
    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()

    if start and task == "classification":
        model_paths = sorted(Path("outputs/models_onnx").glob(f"{dataset}__*.onnx"))
        if not model_paths:
            st.error(f"No ONNX classification models found for dataset '{dataset}'.")
            return
        model_path = st.selectbox("Model", model_paths, format_func=lambda p: p.name)
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        class_names = cfg["data"]["datasets"][dataset]["class_names"]

        if source == "Dataset Stream":
            frame_iter = _iter_dataset_images(dataset, "classification", max_frames)
        else:
            if not video_file:
                st.error("Upload a video file first.")
                return
            frame_iter = _iter_video_frames(video_file.read(), max_frames)

        latencies = []
        for frame, frame_name in frame_iter:
            t0 = time.time()
            inp = _preprocess_classification(frame, cfg["classification"]["input_size"])
            out = sess.run(None, {input_name: inp})[0][0]
            pred = int(np.argmax(out))
            conf = float(np.max(_softmax(out)))
            latency_ms = (time.time() - t0) * 1000.0
            latencies.append(latency_ms)

            annotated = frame.copy()
            cv2.putText(
                annotated,
                f"{class_names[pred]} ({conf:.2f})",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=frame_name, use_container_width=True)
            metrics_placeholder.write(
                {
                    "task": "classification",
                    "dataset": dataset,
                    "profile": profile,
                    "stream_mode": "dataset_stream" if source == "Dataset Stream" else "uploaded_video",
                    "mean_latency_ms": float(np.mean(latencies)),
                    "mean_fps": float(1000.0 / max(np.mean(latencies), 1e-6)),
                }
            )
            time.sleep(max(0.0, (1.0 / max(target_fps, 1)) - (latency_ms / 1000.0)))

    if start and task == "detection":
        run_dirs = []
        for model_name in cfg["detector"]["models"]:
            for base in [Path("outputs/logs"), Path("runs/detect/outputs/logs")]:
                rd = base / f"{dataset}__{model_name}" / "weights"
                if rd.exists():
                    run_dirs.extend(list(rd.glob("*.pt")))
        if not run_dirs:
            st.error(f"No trained YOLO weights found for dataset '{dataset}'.")
            return
        selected_pt = st.selectbox("YOLO Weights", sorted(run_dirs), format_func=lambda p: p.name)
        yolo = YOLO(str(selected_pt))

        if source == "Dataset Stream":
            frame_iter = _iter_dataset_images(dataset, "detection", max_frames)
        else:
            if not video_file:
                st.error("Upload a video file first.")
                return
            frame_iter = _iter_video_frames(video_file.read(), max_frames)

        latencies = []
        for frame, frame_name in frame_iter:
            t0 = time.time()
            res = yolo.predict(source=frame, imgsz=cfg["detector"]["input_size"], device="cpu", verbose=False)
            latency_ms = (time.time() - t0) * 1000.0
            latencies.append(latency_ms)
            plotted = res[0].plot() if res else frame
            frame_placeholder.image(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), caption=frame_name, use_container_width=True)
            metrics_placeholder.write(
                {
                    "task": "detection",
                    "dataset": dataset,
                    "profile": profile,
                    "stream_mode": "dataset_stream" if source == "Dataset Stream" else "uploaded_video",
                    "mean_latency_ms": float(np.mean(latencies)),
                    "mean_fps": float(1000.0 / max(np.mean(latencies), 1e-6)),
                }
            )
            time.sleep(max(0.0, (1.0 / max(target_fps, 1)) - (latency_ms / 1000.0)))


if __name__ == "__main__":
    main()
