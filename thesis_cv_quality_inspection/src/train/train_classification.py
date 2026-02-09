import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.utils.config import load_config
from src.utils.seed import set_seed


def build_model(name: str, num_classes: int):
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model {name}")
    return model


def build_transforms(input_size: int, augment_cfg: dict):
    train_steps = [transforms.Resize((input_size, input_size))]
    if augment_cfg.get("enabled", True):
        hflip_p = float(augment_cfg.get("horizontal_flip", 0.5))
        if hflip_p > 0:
            train_steps.append(transforms.RandomHorizontalFlip(p=hflip_p))

        rotation = float(augment_cfg.get("rotation_deg", 0))
        if rotation > 0:
            train_steps.append(transforms.RandomRotation(degrees=rotation))

        cj = augment_cfg.get("color_jitter", {})
        brightness = float(cj.get("brightness", 0.0))
        contrast = float(cj.get("contrast", 0.0))
        saturation = float(cj.get("saturation", 0.0))
        hue = float(cj.get("hue", 0.0))
        if any(v > 0 for v in [brightness, contrast, saturation, hue]):
            train_steps.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            )

    train_steps.append(transforms.ToTensor())
    eval_steps = [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
    return transforms.Compose(train_steps), transforms.Compose(eval_steps)


def train_one(dataset_dir: Path, model_name: str, cfg):
    input_size = cfg["classification"]["input_size"]
    batch_size = cfg["classification"]["batch_size"]
    epochs = cfg["classification"]["epochs"]
    lr = cfg["classification"]["lr"]
    aug_cfg = cfg["classification"].get("augment", {})
    train_transform, eval_transform = build_transforms(input_size, aug_cfg)

    train_ds = datasets.ImageFolder(dataset_dir / "images" / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(dataset_dir / "images" / "val", transform=eval_transform)

    num_classes = len(train_ds.classes)
    model = build_model(model_name, num_classes)
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_acc = 0.0
    best_path = Path("outputs/checkpoints") / f"{dataset_dir.name}__{model_name}_best.pth"
    history = {"epoch": [], "train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_y_true, best_y_pred = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        train_true, train_pred = [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
            pred = out.argmax(1).cpu().numpy().tolist()
            train_pred.extend(pred)
            train_true.extend(y.cpu().numpy().tolist())

        model.eval()
        y_true, y_pred = [], []
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_losses.append(float(criterion(out, y).item()))
                pred = out.argmax(1).cpu().numpy().tolist()
                y_true.extend(y.numpy().tolist())
                y_pred.extend(pred)

        acc = accuracy_score(y_true, y_pred)
        train_acc = accuracy_score(train_true, train_pred) if train_true else 0.0
        history["epoch"].append(epoch)
        history["train_loss"].append(sum(batch_losses) / max(1, len(batch_losses)))
        history["train_accuracy"].append(float(train_acc))
        history["val_loss"].append(sum(val_losses) / max(1, len(val_losses)))
        history["val_accuracy"].append(float(acc))
        if acc > best_acc:
            best_acc = acc
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
            best_y_true = list(y_true)
            best_y_pred = list(y_pred)

    if not best_y_true:
        best_y_true = list(y_true)
        best_y_pred = list(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(best_y_true, best_y_pred, average="macro", zero_division=0)
    metrics = {
        "dataset": dataset_dir.name,
        "model": model_name,
        "accuracy": float(best_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "class_to_idx": train_ds.class_to_idx,
        "checkpoint": str(best_path),
        "history": history,
    }

    out_hist = Path("outputs/history")
    out_hist.mkdir(parents=True, exist_ok=True)
    (out_hist / f"classification__{dataset_dir.name}__{model_name}.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    fig_dir = Path("docs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Classification Loss - {dataset_dir.name} - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"FIG_Classification_Loss_{dataset_dir.name}__{model_name}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["train_accuracy"], label="train_accuracy")
    plt.plot(history["epoch"], history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Classification Accuracy - {dataset_dir.name} - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"FIG_Classification_AccuracyCurve_{dataset_dir.name}__{model_name}.png", dpi=300)
    plt.close()
    return metrics


def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    out_metrics = Path("outputs/metrics")
    out_metrics.mkdir(parents=True, exist_ok=True)

    for name, dcfg in cfg["data"]["datasets"].items():
        if not dcfg.get("enabled_classification", False):
            continue
        processed_dir = Path(dcfg["processed_dir"])
        if not processed_dir.exists():
            continue
        for model_name in cfg["classification"]["models"]:
            m = train_one(processed_dir, model_name, cfg)
            out_metrics.joinpath(f"classification__{name}__{model_name}.json").write_text(
                json.dumps(m, indent=2),
                encoding="utf-8"
            )


if __name__ == "__main__":
    main()
