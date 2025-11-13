# src/model.py
import torch.nn as nn
import torch.nn.functional as F

class SimpleMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # -> 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # -> 28x28
        self.pool = nn.MaxPool2d(2,2)                 # -> 14x14
        self.dropout1 = nn.Dropout(0.25)
        # corrected flattened size:
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
