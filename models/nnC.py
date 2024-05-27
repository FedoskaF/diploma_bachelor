import torch.nn as nn
import torch.nn.functional as F
from layers.tt_linear import TTLinear

class NetworkC(nn.Module):
    def __init__(self):
        super(NetworkC, self).__init__()
        self.layer1 = TTLinear([1, 7, 7, 7, 1], [7, 4, 4, 4, 7], [8, 8, 8, 8, 4])
        self.layer2 = TTLinear([1, 7, 7, 7, 1], [8, 8, 8, 8, 4], [8, 8, 8, 4, 4])
        self.layer3 = TTLinear([1, 7, 7, 7, 1], [8, 8, 8, 4, 4], [10])

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x
