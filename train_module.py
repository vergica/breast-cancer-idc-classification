import os
import json
import argparse
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

DEFAULT_CONFIG: Dict = {
    "model": {"num_classes": 2, "pretrained": True, "freeze_backbone": False},
    "optimizer": {"name": "adam", "lr": 1e-4, "weight_decay": 1e-5},
    "scheduler": {"name": "cosine", "step_size": 10, "gamma": 0.1, "patience": 5},
    "training": {"epochs": 50, "batch_size": 64, "max_grad_norm": 1.0, "checkpoint_dir": "./checkpoints", "save_every": 10},
    "loss": {"name": "cross_entropy", "alpha": 0.25, "gamma": 2.0},
}


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


def build_criterion(config: Dict) -> nn.Module:
    cfg = config.get("loss", {})
    name = cfg.get("name", "cross_entropy").lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "focal":
        return FocalLoss(alpha=cfg.get("alpha", 0.25), gamma=cfg.get("gamma", 2.0))
    raise ValueError(f"Unsupported loss: {name}")


class BreastResNet(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(config: Dict) -> BreastResNet:
    cfg = config.get("model", {})
    return BreastResNet(
        num_classes=cfg.get("num_classes", 2),
        pretrained=cfg.get("pretrained", True),
        freeze_backbone=cfg.get("freeze_backbone", False),
    )


def build_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    cfg = config.get("optimizer", {})
    name = cfg.get("name", "adam").lower()
    lr = cfg.get("lr", 1e-4)
    wd = cfg.get("weight_decay", 1e-5)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.get("momentum", 0.9), weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict):
    cfg = config.get("scheduler", {})
    name = cfg.get("name", "cosine").lower()
    epochs = config.get("training", {}).get("epochs", 50)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get("step_size", 10), gamma=cfg.get("gamma", 0.1))
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=cfg.get("patience", 5))
    raise ValueError(f"Unsupported scheduler: {name}")


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.best_metric = 0.0
        self.best_epoch = 0
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        max_norm = self.config.get("training", {}).get("max_grad_norm", 1.0)
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.optimizer.step()
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        self.history["train_loss"].append(avg_loss)
        self.history["train_acc"].append(acc)
        return avg_loss, acc

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        avg_loss = total_loss / len(val_loader)
        acc = correct / total
        self.history["val_loss"].append(avg_loss)
        self.history["val_acc"].append(acc)
        return avg_loss, acc

    def save_checkpoint(self, epoch, path, metric):
        torch.save({
            "epoch": epoch, "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": metric, "history": self.history, "config": self.config,
        }, path)

    def save_history(self, checkpoint_dir):
        with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def train(self, train_loader, val_loader, epochs, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Training {epochs} epochs | Device: {self.device} | Checkpoint: {checkpoint_dir}")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)

            if self.config.get("scheduler", {}).get("name") == "plateau":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {lr:.2e}")

            if val_acc > self.best_metric:
                self.best_metric = val_acc
                self.best_epoch = epoch
                self.save_checkpoint(epoch, os.path.join(checkpoint_dir, "best_model.pth"), val_acc)
                print(f"  >> Best model saved (Val Acc: {val_acc:.4f})")

            save_every = self.config.get("training", {}).get("save_every", 10)
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"), val_acc)

            self.save_history(checkpoint_dir)

        print("-" * 60)
        print(f"Done! Best: Epoch {self.best_epoch} Val Acc {self.best_metric:.4f}")
        return self.history


def run_training(data_dir: str, config: Optional[Dict] = None, device: Optional[torch.device] = None):
    from data_module import get_dataloaders

    if config is None:
        config = DEFAULT_CONFIG
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = config.get("training", {}).get("batch_size", 64)
    dataloaders = get_dataloaders(data_dir, batch_size=batch_size)
    train_loader, val_loader = dataloaders["train"], dataloaders["test"]

    model = build_model(config).to(device)
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    trainer = Trainer(model, optimizer, scheduler, criterion, device, config)
    epochs = config.get("training", {}).get("epochs", 50)
    checkpoint_dir = config.get("training", {}).get("checkpoint_dir", "./checkpoints")
    history = trainer.train(train_loader, val_loader, epochs, checkpoint_dir)

    best_path = os.path.join(checkpoint_dir, "best_model.pth")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd", "adamw"])
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "plateau"])
    p.add_argument("--loss", default="cross_entropy", choices=["cross_entropy", "focal"])
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    args = p.parse_args()

    config = {
        "model": {"num_classes": 2, "pretrained": True, "freeze_backbone": args.freeze_backbone},
        "optimizer": {"name": args.optimizer, "lr": args.lr, "weight_decay": 1e-5},
        "scheduler": {"name": args.scheduler},
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "max_grad_norm": args.max_grad_norm, "checkpoint_dir": args.checkpoint_dir, "save_every": 10},
        "loss": {"name": args.loss},
    }
    model, history = run_training(args.data_dir, config=config)
