import torch.nn as nn
import torch.nn.functional as F


class NetworkB(nn.Module):
    def __init__(self):
        super(NetworkB, self).__init__()
        self.layer1 = nn.Linear(784, 16384)
        self.layer2 = nn.Linear(16384, 8192)
        self.layer3 = nn.Linear(8192, 4096)
        self.layer4 = nn.Linear(4096, 2048)
        self.layer5 = nn.Linear(2048, 100)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.softmax(self.layer5(x), dim=1)
        return x
