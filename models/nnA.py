import torch.nn as nn
import torch.nn.functional as F


class NNetworkA(nn.Module):
    def __init__(self):
        super(NetworkA, self).__init__()
        self.layer1 = nn.Linear(784, 16384)
        self.layer2 = nn.Linear(16384, 8192)
        self.layer3 = nn.Linear(8192, 100)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x
