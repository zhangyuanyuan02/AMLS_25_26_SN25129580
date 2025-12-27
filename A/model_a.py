import os
import numpy as np
from medmnist import BreastMNIST
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.ndimage import rotate, gaussian_filter

class TaskA:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data_flag = 'breastmnist'
        # Initialize model (SVM)
        self.model = SVC(kernel='rbf', C=1.0, random_state=42)
        self.scaler = StandardScaler()
        
    def load_data(self):
        """
        load dataset using medmnist BreastMNIST
        """
        print(f"  (Loading {self.data_flag} from {self.dataset_path})...")
        
        # Load datasets
        train_dataset = BreastMNIST(split='train', download=True, root=self.dataset_path)
        val_dataset = BreastMNIST(split='val', download=True, root=self.dataset_path)
        test_dataset = BreastMNIST(split='test', download=True, root=self.dataset_path)
        
        # Extract numpy arrays
        # .imgs: (N, 28, 28)
        # .labels: (N, 1) -> ravel() to (N,)
        X_train, y_train = train_dataset.imgs, train_dataset.labels.ravel()
        X_val, y_val = val_dataset.imgs, val_dataset.labels.ravel()
        X_test, y_test = test_dataset.imgs, test_dataset.labels.ravel()
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def augment_data(self, images, labels):
        """
        Data Augmentation
        1. Rotation
        2. Gaussian Blur
        """
        print("  (Applying Data Augmentation for Classical Model)...")
        aug_images = []
        aug_labels = []

        for img, label in zip(images, labels):
            # Original image
            aug_images.append(img)
            aug_labels.append(label)
            
            # Augmentation 1: Rotate 15 degrees
            rot_img = rotate(img, angle=15, reshape=False)
            aug_images.append(rot_img)
            aug_labels.append(label)
            
            # Augmentation 2: Gaussian Blur
            blur_img = gaussian_filter(img, sigma=1)
            aug_images.append(blur_img)
            aug_labels.append(label)
            
        return np.array(aug_images), np.array(aug_labels)

    def preprocess(self, images, is_training=False):
        """
        Feature processing pipeline:
        1. Flatten (28x28 -> 784)
        2. StandardScaler (Normalization)
        """
        # 1. Flatten
        flat_images = images.reshape(images.shape[0], -1)
        flat_images = flat_images.astype(np.float32)
        
        # 2. Normalize
        if is_training:
            return self.scaler.fit_transform(flat_images)
        else:
            return self.scaler.transform(flat_images)

    def train(self):
        # 1.
        (X_train, y_train), (X_val, y_val), _ = self.load_data()
        
        # 2.
        X_train_aug, y_train_aug = self.augment_data(X_train, y_train)
        
        # 3.
        X_train_processed = self.preprocess(X_train_aug, is_training=True)
        
        # 4. 
        print(f"  (Training SVM with {len(X_train_processed)} samples)...")
        self.model.fit(X_train_processed, y_train_aug)
        
        # 5. 
        train_pred = self.model.predict(X_train_processed)
        acc = accuracy_score(y_train_aug, train_pred)
        print(f"  Task A Training Accuracy: {acc:.4f}")
        return acc

    def test(self):
        # 1. Load Data
        _, _, (X_test, y_test) = self.load_data()
        
        # 2. Preprocess 
        X_test_processed = self.preprocess(X_test, is_training=False)
        
        # 3. Test
        y_pred = self.model.predict(X_test_processed)
        acc = accuracy_score(y_test, y_pred)
        return acc