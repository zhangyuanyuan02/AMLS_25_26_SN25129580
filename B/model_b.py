import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist import BreastMNIST

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class TaskB:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.EPOCHS = 20
        self.BATCH_SIZE = 64
        self.LR = 0.001
        self.model = self.build_model().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)
        
    def build_model(self):
        return CNN()

    def get_transforms(self, mode='train'):
        """
        PyTorch transforms for data augmentation
        """
        if mode == 'train':
            return transforms.Compose([
                transforms.ToTensor(),
                # Augmentation 1: Random Flip
                transforms.RandomHorizontalFlip(p=0.5),
                # Augmentation 2: Random Rotation
                transforms.RandomRotation(degrees=15),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def load_dataloaders(self):
        print(f"  (Loading dataloaders from {self.dataset_path})...")
        
        train_dataset = BreastMNIST(
            split='train', 
            transform=self.get_transforms('train'), 
            download=True, 
            root=self.dataset_path
        )
        
        # test dataset
        test_dataset = BreastMNIST(
            split='test', 
            transform=self.get_transforms('test'), 
            download=True, 
            root=self.dataset_path
        )
        
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        return train_loader, test_loader

    def train(self):
        print(f"  (Training NN on {self.device})...")
        train_loader, _ = self.load_dataloaders()
        
        self.model.train()
        for epoch in range(self.EPOCHS):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device).float()
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
            print(f"    Epoch [{epoch+1}/{self.EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Acc: {correct/total:.4f}")
        acc = correct / total
        return acc

    def test(self):
        _, test_loader = self.load_dataloaders()
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device).float()
                outputs = self.model(inputs)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        acc = correct / total
        return acc