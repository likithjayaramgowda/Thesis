import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import prune
from ultralytics import YOLO

from src.utils.config import load_config


def _apply_pruning(model: nn.Module, amount: float) -> int:
    pruned_layers = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
            pruned_layers += 1
    return pruned_layers


def _build_model(name: str, num_classes: int) -> nn.Module:
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


def _prune_classification(cfg, log, amount: float):
    out_dir = Path("outputs/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, dcfg in cfg["data"]["datasets"].items():
        processed_dir = Path(dcfg["processed_dir"])
        if not processed_dir.exists():
            continue
        num_classes = len(dcfg["class_names"])
        for model_name in cfg["classification"]["models"]:
            ckpt = out_dir / f"{processed_dir.name}__{model_name}_best.pth"
            if not ckpt.exists():
                continue
            model = _build_model(model_name, num_classes)
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model.eval()
            pruned_layers = _apply_pruning(model, amount)
            out_ckpt = out_dir / f"{processed_dir.name}__{model_name}_pruned.pth"
            torch.save(model.state_dict(), out_ckpt)
            log["classification_pruned"].append(
                {
                    "dataset": name,
                    "model": model_name,
                    "layers_pruned": pruned_layers,
                    "checkpoint": str(out_ckpt.as_posix()),
                }
            )


def _prune_detection(cfg, log, amount: float):
    for name, dcfg in cfg["data"]["datasets"].items():
        if not dcfg.get("enabled_detection", False):
            continue
        for model_name in cfg["detector"]["models"]:
            run_dir = Path("outputs/logs") / f"{name}__{model_name}"
            if not run_dir.exists():
                alt = Path("runs/detect/outputs/logs") / f"{name}__{model_name}"
                if alt.exists():
                    run_dir = alt
            best_pt = run_dir / "weights" / "best.pt"
            if not best_pt.exists():
                continue
            try:
                yolo = YOLO(str(best_pt))
                module = yolo.model.model if hasattr(yolo.model, "model") else yolo.model
                pruned_layers = _apply_pruning(module, amount)
                pruned_pt = run_dir / "weights" / "best_pruned.pt"
                yolo.save(str(pruned_pt))
                log["detection_pruned"].append(
                    {
                        "dataset": name,
                        "model": model_name,
                        "layers_pruned": pruned_layers,
                        "checkpoint": str(pruned_pt.as_posix()),
                    }
                )
            except Exception as e:
                log["detection_pruned"].append(
                    {
                        "dataset": name,
                        "model": model_name,
                        "error": str(e),
                    }
                )


def main():
    cfg = load_config()
    pruning_cfg = cfg.get("pruning", {})
    if not pruning_cfg.get("enabled", True):
        return
    amount = float(pruning_cfg.get("amount", 0.3))

    log = {"amount": amount, "classification_pruned": [], "detection_pruned": []}
    _prune_classification(cfg, log, amount)
    _prune_detection(cfg, log, amount)

    log_path = Path("outputs/logs")
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "pruning_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
