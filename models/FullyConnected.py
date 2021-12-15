import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        output = self.fc3(x)

        return output
