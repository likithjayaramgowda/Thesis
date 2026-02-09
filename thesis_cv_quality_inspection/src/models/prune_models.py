import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.nn.utils import prune
from torchvision import datasets, transforms
from ultralytics import YOLO

from src.utils.config import load_config


def _layer_prune_amount(name: str, base_amount: float, max_layer_amount: float, head_amount: float) -> float:
    amount = min(base_amount, max_layer_amount)
    lname = name.lower()
    if "classifier" in lname or lname.endswith(".fc") or ".fc." in lname:
        amount = min(amount, head_amount)
    return float(max(0.0, min(amount, 0.95)))


def _apply_pruning(model: nn.Module, amount: float, max_layer_amount: float, head_amount: float):
    pruned_layers = 0
    sparsities = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_amount = _layer_prune_amount(name, amount, max_layer_amount, head_amount)
            if layer_amount <= 0:
                continue
            prune.l1_unstructured(module, name="weight", amount=layer_amount)
            w = module.weight.detach()
            sparsity = float((w == 0).float().mean().item())
            prune.remove(module, "weight")
            pruned_layers += 1
            sparsities.append(sparsity)
    mean_sparsity = float(sum(sparsities) / len(sparsities)) if sparsities else 0.0
    return pruned_layers, mean_sparsity


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


def _make_cls_loaders(processed_dir: Path, input_size: int, batch_size: int):
    train_root = processed_dir / "images" / "train"
    val_root = processed_dir / "images" / "val"
    if not train_root.exists() or not val_root.exists():
        return None, None

    tf = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])
    train_ds = datasets.ImageFolder(train_root, transform=tf)
    val_ds = datasets.ImageFolder(val_root, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _eval_acc(model: nn.Module, data_loader, device: torch.device, max_batches: int = 0) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            if max_batches > 0 and (i + 1) >= max_batches:
                break
    return float(correct / total) if total else 0.0


def _finetune(model: nn.Module, data_loader, device: torch.device, epochs: int, lr: float, max_batches: int = 0):
    if epochs <= 0:
        return
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            if max_batches > 0 and (i + 1) >= max_batches:
                break


def _prune_classification(cfg, log, amount: float, max_layer_amount: float, head_amount: float):
    out_dir = Path("outputs/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    input_size = int(cfg["classification"]["input_size"])
    finetune_epochs = int(cfg.get("pruning", {}).get("finetune_epochs", 0))
    finetune_lr = float(cfg.get("pruning", {}).get("finetune_lr", 3e-4))
    finetune_batch = int(cfg.get("pruning", {}).get("finetune_batch_size", cfg["classification"]["batch_size"]))
    max_drop = float(cfg.get("pruning", {}).get("max_accuracy_drop", 0.03))
    finetune_max_batches = int(cfg.get("pruning", {}).get("finetune_max_batches", 0))
    eval_max_batches = int(cfg.get("pruning", {}).get("eval_max_batches", 0))
    device = torch.device("cpu")

    for name, dcfg in cfg["data"]["datasets"].items():
        processed_dir = Path(dcfg["processed_dir"])
        if not processed_dir.exists():
            continue
        train_loader, val_loader = _make_cls_loaders(processed_dir, input_size, finetune_batch)
        if train_loader is None or val_loader is None:
            continue

        num_classes = len(dcfg["class_names"])
        for model_name in cfg["classification"]["models"]:
            ckpt = out_dir / f"{processed_dir.name}__{model_name}_best.pth"
            if not ckpt.exists():
                continue

            model = _build_model(model_name, num_classes)
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model.to(device)
            baseline_acc = _eval_acc(model, val_loader, device, max_batches=eval_max_batches)

            pruned_layers, mean_sparsity = _apply_pruning(model, amount, max_layer_amount, head_amount)
            _finetune(
                model,
                train_loader,
                device,
                finetune_epochs,
                finetune_lr,
                max_batches=finetune_max_batches,
            )
            pruned_acc = _eval_acc(model, val_loader, device, max_batches=eval_max_batches)
            acc_drop = float(baseline_acc - pruned_acc)
            accepted = acc_drop <= max_drop

            out_ckpt = out_dir / f"{processed_dir.name}__{model_name}_pruned.pth"
            if accepted:
                torch.save(model.state_dict(), out_ckpt)
            elif out_ckpt.exists():
                out_ckpt.unlink()

            log["classification_pruned"].append(
                {
                    "dataset": name,
                    "model": model_name,
                    "layers_pruned": pruned_layers,
                    "mean_sparsity": mean_sparsity,
                    "baseline_val_accuracy": baseline_acc,
                    "pruned_val_accuracy": pruned_acc,
                    "accuracy_drop": acc_drop,
                    "max_allowed_drop": max_drop,
                    "accepted": accepted,
                    "checkpoint": str(out_ckpt.as_posix()) if accepted else "",
                }
            )


def _prune_detection(cfg, log, amount: float, max_layer_amount: float, head_amount: float):
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
                pruned_layers, mean_sparsity = _apply_pruning(module, amount, max_layer_amount, head_amount)
                pruned_pt = run_dir / "weights" / "best_pruned.pt"
                yolo.save(str(pruned_pt))
                log["detection_pruned"].append(
                    {
                        "dataset": name,
                        "model": model_name,
                        "layers_pruned": pruned_layers,
                        "mean_sparsity": mean_sparsity,
                        "accepted": True,
                        "checkpoint": str(pruned_pt.as_posix()),
                    }
                )
            except Exception as e:
                log["detection_pruned"].append(
                    {
                        "dataset": name,
                        "model": model_name,
                        "error": str(e),
                        "accepted": False,
                    }
                )


def main():
    cfg = load_config()
    pruning_cfg = cfg.get("pruning", {})
    if not pruning_cfg.get("enabled", True):
        return
    amount = float(pruning_cfg.get("amount", 0.3))
    max_layer_amount = float(pruning_cfg.get("max_layer_amount", amount))
    head_amount = float(pruning_cfg.get("classifier_head_amount", amount))

    log = {
        "amount": amount,
        "max_layer_amount": max_layer_amount,
        "classifier_head_amount": head_amount,
        "classification_pruned": [],
        "detection_pruned": [],
    }
    if bool(pruning_cfg.get("prune_classification", True)):
        _prune_classification(cfg, log, amount, max_layer_amount, head_amount)
    if bool(pruning_cfg.get("prune_detection", True)):
        _prune_detection(cfg, log, amount, max_layer_amount, head_amount)

    log_path = Path("outputs/logs")
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "pruning_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
