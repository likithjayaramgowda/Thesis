import json
import random
import shutil
from pathlib import Path
import csv
from PIL import Image
import numpy as np

from src.utils.config import load_config
from src.utils.seed import set_seed


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _collect_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def _split_list(items, splits):
    n = len(items)
    n_train = int(n * splits["train"])
    n_val = int(n * splits["val"])
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def _write_csv(path: Path, rows, header):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def preprocess_classification(dataset_name, cfg):
    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])
    _ensure_dir(processed_dir)
    images_dir = processed_dir / "images"
    _ensure_dir(images_dir)

    class_names = cfg["class_names"]
    splits_cfg = load_config()["splits"]

    # determine class folders
    class_roots = []
    for c in class_names:
        if (raw_dir / c).exists():
            class_roots.append((c, raw_dir / c))

    # NEU-CLS mapping
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

    # KSDD anomaly/good layout mapping
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
        # flattened fallback
        if (raw_dir / "images").exists() and (raw_dir / "ground_truth").exists():
            class_roots.append(("defect", raw_dir / "images"))
    # NEU-CLS layout fallback
    if not class_roots:
        for split_root in [raw_dir / "train" / "images", raw_dir / "validation" / "images"]:
            for c in class_names:
                if (split_root / c).exists():
                    class_roots.append((c, split_root / c))
        if dataset_name == "neu" and not class_roots:
            for split_root in [raw_dir / "train" / "images", raw_dir / "validation" / "images"]:
                mapping = {
                    "crazing": "Cr",
                    "inclusion": "In",
                    "patches": "Pa",
                    "pitted_surface": "PS",
                    "rolled-in_scale": "RS",
                    "scratches": "Sc",
                }
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


def _mask_to_bbox(mask_path: Path):
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return x1, y1, x2, y2, arr.shape[1], arr.shape[0]


def preprocess_detection(dataset_name, cfg, log):
    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])
    _ensure_dir(processed_dir)
    # YOLO expects labels in a "labels/" folder mirroring images/
    labels_dir = processed_dir / "labels"
    images_dir = processed_dir / "images"
    # clean stale detection outputs to avoid unlabeled images
    if labels_dir.exists():
        shutil.rmtree(labels_dir, ignore_errors=True)
    if images_dir.exists():
        shutil.rmtree(images_dir, ignore_errors=True)
    _ensure_dir(labels_dir)
    _ensure_dir(images_dir)

    if dataset_name != "ksdd":
        log["excluded_detection"].append(
            {"dataset": dataset_name, "reason": "No bbox/mask labels available"}
        )
        return

    # Prefer top-level KSDD-only layout if present
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
        log["excluded_detection"].append(
            {"dataset": dataset_name, "reason": "Mask directory not found for bbox derivation"}
        )
        return

    images = _collect_images(img_root)
    # if KSDD SDD structure, restrict to anomaly images for detection
    if (base / "test" / "anomaly").exists():
        images = _collect_images(base / "test" / "anomaly")

    def resolve_mask_path(img_path: Path):
        # try direct mapping
        candidates = [
            mask_root / (img_path.stem + img_path.suffix),
            mask_root / (img_path.stem + ".png"),
            mask_root / (img_path.stem + "_label.png"),
        ]
        # common KSDD layout: ground_truth/anomaly/*.png
        if (mask_root / "anomaly").exists():
            candidates.extend([
                mask_root / "anomaly" / (img_path.stem + ".png"),
                mask_root / "anomaly" / (img_path.stem + "_label.png"),
            ])
        # KSDD SDD: ground_truth/anomaly mirrors test/anomaly
        rel = None
        try:
            rel = img_path.relative_to(base)
        except Exception:
            rel = None
        if rel:
            cand = mask_root / rel
            candidates.append(cand)
            candidates.append(cand.with_suffix(".png"))
            if (mask_root / "anomaly").exists():
                candidates.append(mask_root / "anomaly" / img_path.name)
                candidates.append((mask_root / "anomaly" / img_path.name).with_suffix(".png"))
        for c in candidates:
            if c.exists():
                return c
        return None

    labeled = []
    for p in images:
        mask_path = resolve_mask_path(p)
        if not mask_path:
            continue
        bbox = _mask_to_bbox(mask_path)
        if not bbox:
            continue
        labeled.append((p, bbox))

    if not labeled:
        log["excluded_detection"].append(
            {"dataset": dataset_name, "reason": "No masks with defects found for bbox derivation"}
        )
        return

    splits_cfg = load_config()["splits"]
    random.shuffle(labeled)
    train, val, test = _split_list(labeled, splits_cfg)

    def write_label(img_path: Path, bbox, split: str):
        x1, y1, x2, y2, mw, mh = bbox
        # Normalize using image size; if mask size differs, scale coords.
        img = Image.open(img_path)
        iw, ih = img.size
        if mw != iw or mh != ih:
            sx = iw / float(mw)
            sy = ih / float(mh)
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy
        xc = (x1 + x2) / 2.0 / iw
        yc = (y1 + y2) / 2.0 / ih
        bw = (x2 - x1) / iw
        bh = (y2 - y1) / ih
        label_path = labels_dir / split / "defect" / f"{img_path.stem}.txt"
        _ensure_dir(label_path.parent)
        label_path.write_text(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}", encoding="utf-8")

    for split_name, items in [("train", train), ("val", val), ("test", test)]:
        for p, bbox in items:
            dest_dir = images_dir / split_name / "defect"
            _ensure_dir(dest_dir)
            dest = dest_dir / p.name
            shutil.copy2(p, dest)
            write_label(p, bbox, split_name)

    # copy labels to labels_detection_yolo/ for required format naming
    labels_detection_dir = processed_dir / "labels_detection_yolo"
    _ensure_dir(labels_detection_dir)
    for p in labels_dir.rglob("*.txt"):
        rel = p.relative_to(labels_dir)
        dest = labels_detection_dir / rel
        _ensure_dir(dest.parent)
        dest.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    # write index file for compliance with required format
    label_index = processed_dir / "labels_detection_yolo.txt"
    label_paths = [str(p.as_posix()) for p in labels_dir.rglob("*.txt")]
    label_index.write_text("\n".join(label_paths), encoding="utf-8")


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
            log["excluded_detection"].append(
                {"dataset": name, "reason": "Detection disabled in config"}
            )

    out_log = Path("outputs/logs")
    _ensure_dir(out_log)
    (out_log / "data_prep_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
