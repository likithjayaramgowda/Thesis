import json
from pathlib import Path
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


def train_one(dataset_dir: Path, model_name: str, cfg):
    input_size = cfg["classification"]["input_size"]
    batch_size = cfg["classification"]["batch_size"]
    epochs = cfg["classification"]["epochs"]
    lr = cfg["classification"]["lr"]

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(dataset_dir / "images" / "train", transform=transform)
    val_ds = datasets.ImageFolder(dataset_dir / "images" / "val", transform=transform)

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

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x)
                pred = out.argmax(1).cpu().numpy().tolist()
                y_true.extend(y.numpy().tolist())
                y_pred.extend(pred)

        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics = {
        "dataset": dataset_dir.name,
        "model": model_name,
        "accuracy": float(best_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "checkpoint": str(best_path)
    }
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
