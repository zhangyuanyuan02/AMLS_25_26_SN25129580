import os
from pathlib import Path
from typing import Optional, Union
import numpy as np
from medmnist import BreastMNIST
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.ndimage import rotate, gaussian_filter


class TaskA:
    """
    Task A: Classical ML model (Kernel SVM) for BreastMNIST.

    Supports:
      - Feature pipeline comparison: raw (flatten) vs processed (flatten + standardization)
      - Optional data augmentation (rotation + Gaussian blur)
      - Simple training-budget control via subsampling the training set
      - Auto-download option for local runs (download only if data file missing)
    """

    def __init__(
        self,
        dataset_path,
        svm_C: float = 1.0,
        svm_gamma="scale",
        use_augmentation: bool = True,
        feature_mode: str = "processed",
        train_budget: Optional[float] = None,
        download: Union[bool, str] = "auto",
        random_state: int = 42,
    ):
        self.dataset_path = str(dataset_path)
        self.data_flag = "breastmnist"

        self.use_augmentation = use_augmentation
        self.feature_mode = feature_mode  # "raw" or "processed"
        self.train_budget = train_budget
        self.download = download
        self.random_state = random_state

        # Model (Kernel SVM)
        self.model = SVC(kernel="rbf", C=float(svm_C), gamma=svm_gamma, random_state=random_state)
        self.scaler = StandardScaler()

    # ---------------------- Data ----------------------
    def _should_download(self) -> bool:
        if isinstance(self.download, bool):
            return self.download

        # auto: download only if breastmnist.npz is missing under dataset_path
        root = Path(self.dataset_path)
        return not (root / "breastmnist.npz").exists()

    def load_data(self):
        """
        Load BreastMNIST using medmnist.
        """
        print(f"  (Loading {self.data_flag} from {self.dataset_path})...")

        download_flag = self._should_download()

        train_dataset = BreastMNIST(split="train", download=download_flag, root=self.dataset_path)
        val_dataset = BreastMNIST(split="val", download=download_flag, root=self.dataset_path)
        test_dataset = BreastMNIST(split="test", download=download_flag, root=self.dataset_path)

        X_train, y_train = train_dataset.imgs, train_dataset.labels.ravel()
        X_val, y_val = val_dataset.imgs, val_dataset.labels.ravel()
        X_test, y_test = test_dataset.imgs, test_dataset.labels.ravel()
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    # ---------------------- Augmentation ----------------------
    def augment_data(self, images, labels):
        """
        Simple augmentation for classical features:
          1) rotation (+15 degrees)
          2) Gaussian blur (sigma=1)

        Note: Only used if self.use_augmentation is True.
        """
        if not self.use_augmentation:
            return images, labels

        print("  (Applying Data Augmentation for Classical Model)...")
        aug_images = []
        aug_labels = []

        for img, label in zip(images, labels):
            aug_images.append(img)
            aug_labels.append(label)

            rot_img = rotate(img, angle=15, reshape=False)
            aug_images.append(rot_img)
            aug_labels.append(label)

            blur_img = gaussian_filter(img, sigma=1)
            aug_images.append(blur_img)
            aug_labels.append(label)

        return np.array(aug_images), np.array(aug_labels)

    # ---------------------- Features ----------------------
    def preprocess(self, images, is_training=False, feature_mode: Optional[str] = None):
        """
        Feature pipeline:
          - raw: flatten only
          - processed: flatten + standardization (StandardScaler)
        """
        mode = (feature_mode or self.feature_mode).lower()
        if mode not in {"raw", "processed"}:
            raise ValueError("feature_mode must be 'raw' or 'processed'")

        flat = images.reshape(images.shape[0], -1).astype(np.float32)

        if mode == "raw":
            return flat

        # processed
        if is_training:
            return self.scaler.fit_transform(flat)
        return self.scaler.transform(flat)

    def _apply_train_budget(self, X, y):
        """
        Optional subsampling to simulate training-budget constraints.
        train_budget in (0,1] keeps that fraction of training samples.
        """
        if self.train_budget is None:
            return X, y
        frac = float(self.train_budget)
        frac = max(0.0, min(1.0, frac))
        if frac <= 0:
            raise ValueError("train_budget must be > 0 if provided.")
        n = X.shape[0]
        k = max(1, int(round(n * frac)))
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n, size=k, replace=False)
        return X[idx], y[idx]

    # ---------------------- Metrics ----------------------
    @staticmethod
    def compute_metrics(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    # ---------------------- Train/Test ----------------------
    def train(self, feature_mode: Optional[str] = None):
        (X_train, y_train), _, _ = self.load_data()

        # optional budget
        X_train, y_train = self._apply_train_budget(X_train, y_train)

        # optional augmentation (classical augmentation done on images)
        X_train_aug, y_train_aug = self.augment_data(X_train, y_train)

        # preprocess features
        X_train_processed = self.preprocess(X_train_aug, is_training=True, feature_mode=feature_mode)

        # train
        print("  (Training Kernel SVM)...")
        self.model.fit(X_train_processed, y_train_aug)

        # train metrics
        train_pred = self.model.predict(X_train_processed)
        metrics = self.compute_metrics(y_train_aug, train_pred)
        print(
            "  Task A Train Metrics: "
            f"Acc={metrics['accuracy']:.4f}, "
            f"P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )
        return metrics

    def test(self, feature_mode: Optional[str] = None):
        _, _, (X_test, y_test) = self.load_data()

        X_test_processed = self.preprocess(X_test, is_training=False, feature_mode=feature_mode)
        y_pred = self.model.predict(X_test_processed)
        return self.compute_metrics(y_test, y_pred)

    def compare_feature_pipelines(self):
        """
        Requirement-oriented helper:
        trains/evaluates two pipelines and returns their test metrics:
          - raw: flatten only
          - processed: flatten + standardization
        Note: This re-trains the model for each pipeline.
        """
        results = {}
        for mode in ["raw", "processed"]:
            # rebuild model+scaler each run to avoid leakage
            self.model = SVC(kernel="rbf", C=self.model.C, gamma=self.model.gamma, random_state=self.random_state)
            self.scaler = StandardScaler()
            print(f"\n  [Task A] Training with feature_mode='{mode}'")
            self.train(feature_mode=mode)
            results[mode] = self.test(feature_mode=mode)
        return results
