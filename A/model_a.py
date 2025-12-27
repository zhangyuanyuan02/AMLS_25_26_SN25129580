import os
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# import medmnist

class TaskA:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None # such as SVM
        
    def load_data(self):
        # load dataset from MNIST
        print(f"  (Loading data from {self.dataset_path})...")
        pass

    def preprocess(self, images):
        """
        # feature extraction example: flatten images, or data ehnancement
        """
        # flat_images = images.reshape(images.shape[0], -1)
        # return flat_images
        pass

    def train(self):
        # 1. Load Data
        # 2. Preprocess
        # 3. Augmentation
        # 4. Fit Model (e.g., self.model.fit(X_train, y_train))
        return 

    def test(self):
        return 