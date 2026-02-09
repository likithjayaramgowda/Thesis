import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from src.utils.config import load_config
from src.utils.seed import set_seed


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _collect_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def _split_list(items, splits):
    n = len(items)
    n_train = int(n * splits["train"])
    n_val = int(n * splits["val"])
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test


def _split_counts(n: int, splits):
    if n <= 0:
        return 0, 0, 0
    ratios = np.array(
        [float(splits.get("train", 0.7)), float(splits.get("val", 0.15)), float(splits.get("test", 0.15))],
        dtype=np.float64,
    )
    if ratios.sum() <= 0:
        ratios = np.array([0.7, 0.15, 0.15], dtype=np.float64)
    ratios = ratios / ratios.sum()
    raw = ratios * n
    counts = np.floor(raw).astype(int)
    remainder = int(n - counts.sum())
    if remainder > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            counts[idx] += 1

    # Keep validation/test non-empty for reasonably sized strata.
    if n >= 5:
        for idx in [1, 2]:
            if counts[idx] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[idx] += 1

    # Ensure train split is non-empty.
    if counts[0] == 0:
        if n == 1:
            counts = np.array([1, 0, 0], dtype=int)
        elif n == 2:
            counts = np.array([1, 0, 1], dtype=int)
        else:
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[0] += 1

    return int(counts[0]), int(counts[1]), int(counts[2])


def _split_detection_records(records, splits, seed: int):
    rng = random.Random(seed)
    split_to_records = {"train": [], "val": [], "test": []}
    strata = {}
    for r in records:
        key = (r.get("class_name", "defect"), bool(r.get("bbox")))
        strata.setdefault(key, []).append(r)

    for _, items in strata.items():
        items = list(items)
        rng.shuffle(items)
        n_train, n_val, n_test = _split_counts(len(items), splits)
        split_to_records["train"].extend(items[:n_train])
        split_to_records["val"].extend(items[n_train : n_train + n_val])
        split_to_records["test"].extend(items[n_train + n_val : n_train + n_val + n_test])

    for split_name in split_to_records:
        rng.shuffle(split_to_records[split_name])
    return split_to_records


def _balance_background_records(records, max_ratio: float, seed: int):
    positives = [r for r in records if r.get("bbox") is not None]
    negatives = [r for r in records if r.get("bbox") is None]
    if not positives or not negatives or max_ratio <= 0:
        return records
    max_negatives = int(len(positives) * max_ratio)
    if len(negatives) <= max_negatives:
        return records
    rng = random.Random(seed)
    rng.shuffle(negatives)
    return positives + negatives[:max_negatives]


def _write_csv(path: Path, rows, header):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _mask_to_bbox(mask_path: Path):
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return x1, y1, x2, y2, arr.shape[1], arr.shape[0]


def _write_yolo_label(label_path: Path, class_id: int, bbox, img_size):
    x1, y1, x2, y2 = bbox
    iw, ih = img_size
    xc = (x1 + x2) / 2.0 / iw
    yc = (y1 + y2) / 2.0 / ih
    bw = (x2 - x1) / iw
    bh = (y2 - y1) / ih
    _ensure_dir(label_path.parent)
    label_path.write_text(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}", encoding="utf-8")


def _write_empty_yolo_label(label_path: Path):
    _ensure_dir(label_path.parent)
    label_path.write_text("", encoding="utf-8")


def _unique_image_destination(img_dest_dir: Path, img_src: Path):
    direct = img_dest_dir / img_src.name
    if not direct.exists():
        return direct
    digest = hashlib.sha1(str(img_src.resolve()).encode("utf-8")).hexdigest()[:10]
    candidate = img_dest_dir / f"{img_src.stem}_{digest}{img_src.suffix}"
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        retry = img_dest_dir / f"{img_src.stem}_{digest}_{idx}{img_src.suffix}"
        if not retry.exists():
            return retry
        idx += 1


def _write_data_yaml(processed_dir: Path, class_names):
    data_yaml = processed_dir / "yolo_data.yaml"
    names_serialized = "[" + ", ".join([f"'{c}'" for c in class_names]) + "]"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {processed_dir.resolve().as_posix()}",
                "train: yolo/images/train",
                "val: yolo/images/val",
                "test: yolo/images/test",
                f"names: {names_serialized}",
            ]
        ),
        encoding="utf-8",
    )


def preprocess_classification(dataset_name, cfg):
    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])
    _ensure_dir(processed_dir)
    images_dir = processed_dir / "images"
    _ensure_dir(images_dir)

    class_names = cfg["class_names"]
    splits_cfg = load_config()["splits"]

    class_roots = []
    for c in class_names:
        if (raw_dir / c).exists():
            class_roots.append((c, raw_dir / c))

    if dataset_name == "neu" and not class_roots:
        mapping = {
            "crazing": "Cr",
            "inclusion": "In",
            "patches": "Pa",
            "pitted_surface": "PS",
            "rolled-in_scale": "RS",
            "scratches": "Sc",
        }
        for src, dst in mapping.items():
            if (raw_dir / src).exists():
                class_roots.append((dst, raw_dir / src))

    if dataset_name == "ksdd" and not class_roots:
        nested = raw_dir / "content" / "dataset" / "mvisa" / "data" / "ksdd" / "SDD"
        base = nested if nested.exists() else raw_dir
        defect_dir = base / "test" / "anomaly"
        good_dirs = [base / "train" / "good", base / "test" / "good"]
        if defect_dir.exists():
            class_roots.append(("defect", defect_dir))
        for gd in good_dirs:
            if gd.exists():
                class_roots.append(("normal", gd))
        if (raw_dir / "images").exists() and (raw_dir / "ground_truth").exists():
            class_roots.append(("defect", raw_dir / "images"))

    if not class_roots:
        for split_root in [raw_dir / "train" / "images", raw_dir / "validation" / "images"]:
            for c in class_names:
                if (split_root / c).exists():
                    class_roots.append((c, split_root / c))
        if dataset_name == "neu" and not class_roots:
            mapping = {
                "crazing": "Cr",
                "inclusion": "In",
                "patches": "Pa",
                "pitted_surface": "PS",
                "rolled-in_scale": "RS",
                "scratches": "Sc",
            }
            for split_root in [raw_dir / "train" / "images", raw_dir / "validation" / "images"]:
                for src, dst in mapping.items():
                    if (split_root / src).exists():
                        class_roots.append((dst, split_root / src))

    rows_labels = []
    rows_splits = []

    for class_name, class_dir in class_roots:
        imgs = _collect_images(class_dir)
        random.shuffle(imgs)
        train, val, test = _split_list(imgs, splits_cfg)
        for split_name, items in [("train", train), ("val", val), ("test", test)]:
            split_dir = images_dir / split_name / class_name
            _ensure_dir(split_dir)
            for p in items:
                dest = _unique_image_destination(split_dir, p)
                shutil.copy2(p, dest)
                rel = dest.as_posix()
                rows_labels.append([rel, class_name])
                rows_splits.append([rel, split_name])

    _write_csv(processed_dir / "labels_classification.csv", rows_labels, ["image_path", "class"])
    _write_csv(processed_dir / "splits.csv", rows_splits, ["image_path", "split"])


def _center_box(iw: int, ih: int, frac: float):
    frac = float(np.clip(frac, 0.05, 1.0))
    bw = max(2, int(iw * frac))
    bh = max(2, int(ih * frac))
    x1 = max(0, (iw - bw) // 2)
    y1 = max(0, (ih - bh) // 2)
    x2 = min(iw - 1, x1 + bw - 1)
    y2 = min(ih - 1, y1 + bh - 1)
    return float(x1), float(y1), float(x2), float(y2)


def _weak_bbox_from_image(img_path: Path, weak_cfg):
    quantile = float(weak_cfg.get("quantile", 0.98))
    quantile = float(np.clip(quantile, 0.80, 0.999))
    trim = float(weak_cfg.get("trim", 0.15))
    trim = float(np.clip(trim, 0.01, 0.3))
    blur_radius = float(weak_cfg.get("blur_radius", 2.0))
    min_side_frac = float(weak_cfg.get("min_side_frac", 0.15))
    max_side_frac = float(weak_cfg.get("max_side_frac", 0.75))
    fallback_frac = float(weak_cfg.get("fallback_box_frac", 0.5))

    img_gray = Image.open(img_path).convert("L")
    iw, ih = img_gray.size
    arr = np.array(img_gray, dtype=np.float32) / 255.0
    smooth = np.array(img_gray.filter(ImageFilter.GaussianBlur(radius=blur_radius)), dtype=np.float32) / 255.0
    score = np.abs(arr - smooth)
    thr = float(np.quantile(score, quantile))
    ys, xs = np.where(score >= thr)
    if len(xs) < 16 or len(ys) < 16:
        return _center_box(iw, ih, fallback_frac)

    x1 = float(np.quantile(xs, trim))
    x2 = float(np.quantile(xs, 1.0 - trim))
    y1 = float(np.quantile(ys, trim))
    y2 = float(np.quantile(ys, 1.0 - trim))
    if x2 <= x1 or y2 <= y1:
        return _center_box(iw, ih, fallback_frac)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    cur_w = max(1.0, (x2 - x1 + 1.0))
    cur_h = max(1.0, (y2 - y1 + 1.0))
    min_w = max(4.0, iw * min_side_frac)
    min_h = max(4.0, ih * min_side_frac)
    max_w = max(min_w, iw * max_side_frac)
    max_h = max(min_h, ih * max_side_frac)
    out_w = float(np.clip(cur_w, min_w, max_w))
    out_h = float(np.clip(cur_h, min_h, max_h))
    x1 = max(0.0, cx - out_w / 2.0)
    y1 = max(0.0, cy - out_h / 2.0)
    x2 = min(float(iw - 1), x1 + out_w - 1.0)
    y2 = min(float(ih - 1), y1 + out_h - 1.0)
    return x1, y1, x2, y2


def _index_masks(mask_roots):
    idx = {}
    for root in mask_roots:
        if not root.exists():
            continue
        for p in _collect_images(root):
            idx.setdefault(p.stem, []).append(p)
    return idx


def _preprocess_detection_ksdd(raw_dir: Path):
    nested = raw_dir / "content" / "dataset" / "mvisa" / "data" / "ksdd" / "SDD"
    image_roots = []
    for cand in [
        raw_dir / "images",
        nested / "train" / "good",
        nested / "test" / "good",
        nested / "test" / "anomaly",
    ]:
        if cand.exists():
            image_roots.append(cand)
    mask_roots = [raw_dir / "ground_truth", nested / "ground_truth"]
    mask_index = _index_masks(mask_roots)

    seen = set()
    records = []
    for root in image_roots:
        for p in _collect_images(root):
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            img = Image.open(p)
            bbox = None
            for mask_path in mask_index.get(p.stem, []):
                maybe = _mask_to_bbox(mask_path)
                if maybe:
                    bbox = maybe[:4]
                    break
            records.append({"img_path": p, "class_name": "defect", "bbox": bbox, "img_size": img.size})
    return records


def _preprocess_detection_dagm(raw_dir: Path):
    images = _collect_images(raw_dir)
    records = []
    for p in images:
        if "_label" in p.stem.lower():
            continue
        candidates = [
            p.parent / f"{p.stem}_label{p.suffix}",
            p.parent / f"{p.stem}_label.png",
            p.parent / "Label" / f"{p.stem}_label{p.suffix}",
            p.parent / "Label" / f"{p.stem}_label.PNG",
            p.parent / "Label" / f"{p.stem}_label.png",
        ]
        mask_path = None
        for c in candidates:
            if c.exists():
                mask_path = c
                break
        img = Image.open(p)
        bbox = None
        if mask_path:
            maybe_bbox = _mask_to_bbox(mask_path)
            if maybe_bbox:
                bbox = maybe_bbox[:4]
        records.append({"img_path": p, "class_name": "defect", "bbox": bbox, "img_size": img.size})
    return records


def _preprocess_detection_neu(processed_dir: Path, weak_cfg):
    labels_path = processed_dir / "labels_classification.csv"
    splits_path = processed_dir / "splits.csv"
    if not labels_path.exists() or not splits_path.exists():
        return []
    labels = {}
    with labels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["image_path"]] = row["class"]
    rows = []
    with splits_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = Path(row["image_path"])
            if not img_path.exists():
                continue
            class_name = labels.get(row["image_path"])
            if not class_name:
                continue
            img = Image.open(img_path)
            iw, ih = img.size
            bbox = _weak_bbox_from_image(img_path, weak_cfg)
            rows.append(
                {
                    "img_path": img_path,
                    "class_name": class_name,
                    "bbox": bbox,
                    "img_size": (iw, ih),
                }
            )
    return rows


def preprocess_detection(dataset_name, cfg, log):
    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])
    _ensure_dir(processed_dir)
    yolo_root = processed_dir / "yolo"
    labels_dir = yolo_root / "labels"
    images_dir = yolo_root / "images"
    legacy_labels_dir = processed_dir / "labels_detection"
    legacy_images_dir = processed_dir / "images_detection"
    for d in [labels_dir, images_dir, legacy_labels_dir, legacy_images_dir]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    _ensure_dir(labels_dir)
    _ensure_dir(images_dir)

    full_cfg = load_config()
    det_cfg = full_cfg.get("detector", {})
    det_splits = det_cfg.get("splits", full_cfg.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15}))
    max_background_ratio = float(det_cfg.get("max_background_ratio", 2.0))
    weak_cfg = det_cfg.get("neu_weak_label", {})

    if dataset_name == "ksdd":
        records = _preprocess_detection_ksdd(raw_dir)
        class_names = ["defect"]
    elif dataset_name == "dagm":
        records = _preprocess_detection_dagm(raw_dir)
        class_names = ["defect"]
    elif dataset_name == "neu":
        records = _preprocess_detection_neu(processed_dir, weak_cfg)
        class_names = list(cfg["class_names"])
    else:
        records = []
        class_names = ["defect"]

    if not records:
        log["excluded_detection"].append(
            {"dataset": dataset_name, "reason": "No detection annotations found (mask or weak-label)."}
        )
        return

    split_seed = int(full_cfg.get("project", {}).get("seed", 42)) + sum(ord(ch) for ch in dataset_name)
    if dataset_name in {"dagm", "ksdd"}:
        records = _balance_background_records(records, max_ratio=max_background_ratio, seed=split_seed)
    split_to_records = _split_detection_records(records, det_splits, split_seed)
    class_to_id = {c: i for i, c in enumerate(class_names)}

    for split_name, items in split_to_records.items():
        for item in items:
            class_name = item["class_name"]
            is_positive = item.get("bbox") is not None
            if is_positive and class_name not in class_to_id:
                continue
            img_src = item["img_path"]
            class_dir = class_name if dataset_name == "neu" else "defect"
            img_dest_dir = images_dir / split_name / class_dir
            _ensure_dir(img_dest_dir)
            img_dest = _unique_image_destination(img_dest_dir, img_src)
            shutil.copy2(img_src, img_dest)

            label_path = labels_dir / split_name / class_dir / f"{img_dest.stem}.txt"
            if is_positive:
                _write_yolo_label(label_path, class_to_id[class_name], item["bbox"], item["img_size"])
            else:
                _write_empty_yolo_label(label_path)

    label_index = processed_dir / "labels_detection_yolo.txt"
    label_paths = [str(p.as_posix()) for p in labels_dir.rglob("*.txt")]
    label_index.write_text("\n".join(label_paths), encoding="utf-8")

    _write_data_yaml(processed_dir, class_names)

    positives = sum(1 for r in records if r.get("bbox") is not None)
    negatives = len(records) - positives
    split_stats = {}
    for split_name, items in split_to_records.items():
        p = sum(1 for r in items if r.get("bbox") is not None)
        split_stats[split_name] = {"total": len(items), "positive": p, "negative": len(items) - p}
    if "detection_stats" not in log:
        log["detection_stats"] = []
    log["detection_stats"].append(
        {
            "dataset": dataset_name,
            "total_records": len(records),
            "positive_records": positives,
            "negative_records": negatives,
            "splits": split_stats,
        }
    )


def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    log = {"excluded_detection": []}

    for name, dcfg in cfg["data"]["datasets"].items():
        raw_dir = Path(dcfg["raw_dir"])
        if not raw_dir.exists():
            continue
        if dcfg.get("enabled_classification", False):
            preprocess_classification(name, dcfg)
        if dcfg.get("enabled_detection", False):
            preprocess_detection(name, dcfg, log)
        else:
            log["excluded_detection"].append({"dataset": name, "reason": "Detection disabled in config"})

    out_log = Path("outputs/logs")
    _ensure_dir(out_log)
    (out_log / "data_prep_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
