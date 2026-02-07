import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

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


def _write_data_yaml(processed_dir: Path, class_names):
    data_yaml = processed_dir / "yolo_data.yaml"
    names_serialized = "[" + ", ".join([f"'{c}'" for c in class_names]) + "]"
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
                dest = split_dir / p.name
                if dest.exists():
                    dest = split_dir / f"{split_name}_{p.name}"
                shutil.copy2(p, dest)
                rel = dest.as_posix()
                rows_labels.append([rel, class_name])
                rows_splits.append([rel, split_name])

    _write_csv(processed_dir / "labels_classification.csv", rows_labels, ["image_path", "class"])
    _write_csv(processed_dir / "splits.csv", rows_splits, ["image_path", "split"])


def _preprocess_detection_ksdd(raw_dir: Path):
    base = raw_dir
    nested = raw_dir / "content" / "dataset" / "mvisa" / "data" / "ksdd" / "SDD"
    top_images = raw_dir / "images"
    top_masks = raw_dir / "ground_truth"
    if not (top_images.exists() and top_masks.exists()) and nested.exists():
        base = nested

    img_root = None
    mask_root = None
    for cand in [base / "images", base / "train", base / "test"]:
        if cand.exists() and _collect_images(cand):
            img_root = cand
            break
    for cand in [base / "ground_truth", base / "masks", base / "labels"]:
        if cand.exists() and _collect_images(cand):
            mask_root = cand
            break
    if not img_root or not mask_root:
        return []

    images = _collect_images(img_root)
    if (base / "test" / "anomaly").exists():
        images = _collect_images(base / "test" / "anomaly")

    labeled = []
    for p in images:
        candidates = [
            mask_root / (p.stem + p.suffix),
            mask_root / (p.stem + ".png"),
            mask_root / (p.stem + "_label.png"),
            mask_root / "anomaly" / (p.stem + ".png"),
            mask_root / "anomaly" / (p.stem + "_label.png"),
        ]
        mask_path = None
        for c in candidates:
            if c.exists():
                mask_path = c
                break
        if not mask_path:
            continue
        bbox = _mask_to_bbox(mask_path)
        if not bbox:
            continue
        img = Image.open(p)
        labeled.append({"img_path": p, "class_name": "defect", "bbox": bbox[:4], "img_size": img.size})
    return labeled


def _preprocess_detection_dagm(raw_dir: Path):
    images = _collect_images(raw_dir)
    labeled = []
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
        if not mask_path:
            continue
        bbox = _mask_to_bbox(mask_path)
        if not bbox:
            continue
        img = Image.open(p)
        labeled.append({"img_path": p, "class_name": "defect", "bbox": bbox[:4], "img_size": img.size})
    return labeled


def _preprocess_detection_neu(processed_dir: Path):
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
            rows.append(
                {
                    "img_path": img_path,
                    "class_name": class_name,
                    "bbox": (0.0, 0.0, float(iw - 1), float(ih - 1)),
                    "img_size": (iw, ih),
                    "preset_split": row["split"],
                }
            )
    return rows


def preprocess_detection(dataset_name, cfg, log):
    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])
    _ensure_dir(processed_dir)
    labels_dir = processed_dir / "labels_detection"
    images_dir = processed_dir / "images_detection"
    if labels_dir.exists():
        shutil.rmtree(labels_dir, ignore_errors=True)
    if images_dir.exists():
        shutil.rmtree(images_dir, ignore_errors=True)
    _ensure_dir(labels_dir)
    _ensure_dir(images_dir)

    if dataset_name == "ksdd":
        records = _preprocess_detection_ksdd(raw_dir)
        class_names = ["defect"]
    elif dataset_name == "dagm":
        records = _preprocess_detection_dagm(raw_dir)
        class_names = ["defect"]
    elif dataset_name == "neu":
        records = _preprocess_detection_neu(processed_dir)
        class_names = list(cfg["class_names"])
    else:
        records = []
        class_names = ["defect"]

    if not records:
        log["excluded_detection"].append(
            {"dataset": dataset_name, "reason": "No detection annotations found (mask or weak-label)."}
        )
        return

    class_to_id = {c: i for i, c in enumerate(class_names)}
    splits_cfg = load_config()["splits"]

    if dataset_name == "neu":
        split_to_records = {"train": [], "val": [], "test": []}
        for r in records:
            split_to_records[r.get("preset_split", "train")].append(r)
    else:
        random.shuffle(records)
        train, val, test = _split_list(records, splits_cfg)
        split_to_records = {"train": train, "val": val, "test": test}

    for split_name, items in split_to_records.items():
        for item in items:
            class_name = item["class_name"]
            if class_name not in class_to_id:
                continue
            img_src = item["img_path"]
            class_dir = class_name if dataset_name == "neu" else "defect"
            img_dest_dir = images_dir / split_name / class_dir
            _ensure_dir(img_dest_dir)
            img_dest = img_dest_dir / img_src.name
            if img_dest.exists():
                img_dest = img_dest_dir / f"{split_name}_{img_src.name}"
            shutil.copy2(img_src, img_dest)

            label_path = labels_dir / split_name / class_dir / f"{img_dest.stem}.txt"
            _write_yolo_label(label_path, class_to_id[class_name], item["bbox"], item["img_size"])

    labels_detection_dir = processed_dir / "labels_detection_yolo"
    _ensure_dir(labels_detection_dir)
    for p in labels_dir.rglob("*.txt"):
        rel = p.relative_to(labels_dir)
        dest = labels_detection_dir / rel
        _ensure_dir(dest.parent)
        dest.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    label_index = processed_dir / "labels_detection_yolo.txt"
    label_paths = [str(p.as_posix()) for p in labels_dir.rglob("*.txt")]
    label_index.write_text("\n".join(label_paths), encoding="utf-8")

    _write_data_yaml(processed_dir, class_names)


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
