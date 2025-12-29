import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist import BreastMNIST


class CNN(nn.Module):
    """
    Simple CNN for BreastMNIST (1x28x28).
    Capacity can be controlled via base_channels.
    """

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 28 -> 14 -> 7 -> 3 (floor)
        self.fc = nn.Linear(c3 * 3 * 3, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class TaskB:
    """
    Task B: Deep Learning model (CNN) for BreastMNIST.

    Supports:
      - Optional data augmentation (flip + rotation)
      - Training budget control (epochs and optional subsampling of train set)
      - Model capacity control (base_channels)
      - Auto-download option for local runs (download only if data file missing)
      - Accuracy/Precision/Recall/F1 evaluation
    """

    def __init__(
        self,
        dataset_path,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        use_augmentation: bool = True,
        base_channels: int = 16,
        train_budget: Optional[float] = None,
        download: Union[bool, str] = "auto",
        seed: int = 42,
    ):
        self.dataset_path = str(dataset_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.EPOCHS = int(epochs)
        self.BATCH_SIZE = int(batch_size)
        self.LR = float(lr)

        self.use_augmentation = use_augmentation
        self.base_channels = int(base_channels)
        self.train_budget = train_budget
        self.download = download

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model = CNN(base_channels=self.base_channels).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)

    # ---------------------- Data ----------------------
    def _should_download(self) -> bool:
        if isinstance(self.download, bool):
            return self.download
        root = Path(self.dataset_path)
        return not (root / "breastmnist.npz").exists()

    def get_transforms(self, mode="train"):
        """
        PyTorch transforms for data augmentation.
        """
        if mode == "train" and self.use_augmentation:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def _apply_train_budget(self, dataset):
        """
        Optional subsampling of train dataset: keep a fraction in (0,1].
        """
        if self.train_budget is None:
            return dataset
        frac = float(self.train_budget)
        frac = max(0.0, min(1.0, frac))
        if frac <= 0:
            raise ValueError("train_budget must be > 0 if provided.")
        n = len(dataset)
        k = max(1, int(round(n * frac)))
        g = torch.Generator()
        g.manual_seed(42)
        subset_indices = torch.randperm(n, generator=g)[:k].tolist()
        return data.Subset(dataset, subset_indices)

    def load_dataloaders(self):
        print(f"  (Loading dataloaders from {self.dataset_path})...")

        download_flag = self._should_download()

        train_dataset = BreastMNIST(
            split="train",
            transform=self.get_transforms("train"),
            download=download_flag,
            root=self.dataset_path,
        )
        test_dataset = BreastMNIST(
            split="test",
            transform=self.get_transforms("test"),
            download=download_flag,
            root=self.dataset_path,
        )

        train_dataset = self._apply_train_budget(train_dataset)

        train_loader = data.DataLoader(
            train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=0
        )
        test_loader = data.DataLoader(
            test_dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=0
        )
        return train_loader, test_loader

    # ---------------------- Metrics ----------------------
    @staticmethod
    def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """
        y_true, y_pred: 0/1 tensors of shape (N,)
        """
        y_true = y_true.int()
        y_pred = y_pred.int()

        tp = int(((y_pred == 1) & (y_true == 1)).sum().item())
        tn = int(((y_pred == 0) & (y_true == 0)).sum().item())
        fp = int(((y_pred == 1) & (y_true == 0)).sum().item())
        fn = int(((y_pred == 0) & (y_true == 1)).sum().item())

        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return {"accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)}

    # ---------------------- Train/Test ----------------------
    def train(self):
        train_loader, _ = self.load_dataloaders()
        self.model.train()

        for epoch in range(self.EPOCHS):
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).float().view(-1, 1)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            avg_loss = running_loss / len(train_loader.dataset)
            print(f"  Epoch [{epoch+1}/{self.EPOCHS}] Loss: {avg_loss:.4f}")

        # Return train metrics for convenience (on train subset)
        self.model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).float().view(-1)
                logits = self.model(inputs).view(-1)
                preds = (torch.sigmoid(logits) > 0.5).int()
                all_true.append(targets.int().cpu())
                all_pred.append(preds.cpu())
        y_true = torch.cat(all_true)
        y_pred = torch.cat(all_pred)
        metrics = self.compute_metrics(y_true, y_pred)
        print(
            "  Task B Train Metrics: "
            f"Acc={metrics['accuracy']:.4f}, "
            f"P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )
        return metrics

    def test(self):
        _, test_loader = self.load_dataloaders()
        self.model.eval()

        all_true, all_pred = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).float().view(-1)
                logits = self.model(inputs).view(-1)
                preds = (torch.sigmoid(logits) > 0.5).int()

                all_true.append(targets.int().cpu())
                all_pred.append(preds.cpu())

        y_true = torch.cat(all_true)
        y_pred = torch.cat(all_pred)
        return self.compute_metrics(y_true, y_pred)
