import torch
import torch.nn as nn
import torch.nn.functional as F

class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(15329, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 8)
        self.fc5 = torch.nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(15329, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 8)
        self.fc4 = torch.nn.Linear(8, 1)
    
    def forward(self, x):
        self.x = x
        self.x = torch.tanh(self.fc1(self.x))
        self.x = torch.tanh(self.fc2(self.x))
        self.x = torch.tanh(self.fc3(self.x))
        self.x = torch.sigmoid(self.fc4(self.x))
        return self.x

class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(15329, 7636)
        self.fc2 = torch.nn.Linear(7636, 3818)
        self.fc3 = torch.nn.Linear(3818, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, 1)
    
    def forward(self, x):
        self.x = x
        self.x = torch.relu(self.fc1(self.x))
        self.x = torch.relu(self.fc2(self.x))
        self.x = torch.relu(self.fc3(self.x))
        self.x = torch.relu(self.fc4(self.x))
        self.x = torch.sigmoid(self.fc5(self.x))
        return self.x