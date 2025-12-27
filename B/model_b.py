import os
# import torch
# import torch.nn as nn
# from torchvision import transforms

class TaskB:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.build_model()
        
    def build_model(self):
        # initialize CNN or ResNet model
        # return MyCNN()
        pass

    def get_transforms(self):
        """
        data augmentation examples:
        1. Rotation/Flip
        2. Noise/Blur/Brightness
        """
        # return transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(10),
        #     transforms.ToTensor()
        # ])
        pass

    def train(self):
        # Implement PyTorch training loop
        # Epochs, Batch size
        return 0.90

    def test(self):
        # Evaluate on Test set
        return 0.88