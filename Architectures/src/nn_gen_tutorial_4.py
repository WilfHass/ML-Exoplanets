import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # (in channels: RBG, out channels, kernel size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) # (in channels, out channels, kernel size)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # output of conv2 is 16*5*5 !
        self.fc2 = nn.Linear(120, 84) # inner/outer dims must match
        self.fc3 = nn.Linear(84, num_classes) # num_classes for CIFAR10 is 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # No need to have softmax the last layer. 
        # We can let the loss function handle this internally
        x = self.fc3(x) 
        return x

    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()