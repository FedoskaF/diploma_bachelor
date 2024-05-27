import torch
import torch.nn as nn
import numpy as np
from layers.tt_conv import TTConv

class VGG16_TTConv(nn.Module):
    def __init__(self):
        super(VGG16_TTConv, self).__init__()
        self.features = nn.Sequential(
            TTConv(window=[3, 3], inp_ch_modes=np.array([1, 3]), out_ch_modes=np.array([8, 8]),
                   ranks=np.array([1, 9, 1]), strides=[1, 1], padding='same'),
            nn.ReLU(inplace=True),
            TTConv(window=[3, 3], inp_ch_modes=np.array([1, 4, 4, 4]), out_ch_modes=np.array([1, 4, 4, 4]),
                   ranks=np.array([1, 9, 9, 9, 1]), strides=[1, 1], padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            TTConv(window=[3, 3], inp_ch_modes=np.array([1, 4, 4, 4]), out_ch_modes=np.array([2, 4, 4, 4]),
                   ranks=np.array([1, 9, 9, 9, 1]), strides=[1, 1], padding='same'),
            nn.ReLU(inplace=True),
            TTConv(window=[3, 3], inp_ch_modes=np.array([2, 4, 4, 4]), out_ch_modes=np.array([2, 4, 4, 4]),
                   ranks=np.array([1, 9, 9, 9, 1]), strides=[1, 1], padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            TTConv(window=[3, 3], inp_ch_modes=np.array([2, 4, 4, 4]), out_ch_modes=np.array([4, 4, 4, 4]),
                   ranks=np.array([1, 9, 9, 9, 1]), strides=[1, 1], padding='same'),
            nn.ReLU(inplace=True),
            TTConv(window=[3, 3], inp_ch_modes=np.array([4, 4, 4, 4]), out_ch_modes=np.array([4, 4, 4, 4]),
                   ranks=np.array([1, 9, 9, 9, 1]), strides=[1, 1], padding='same'),
            nn.ReLU(inplace=True),
            TTConv(window=[3, 3], inp_ch_modes=np.array([4, 4, 4, 4]), out_ch_modes=np.array([4, 4, 4, 4]),
                   ranks=np.array([1, 9, 9, 9, 1]), strides=[1, 1], padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            TTConv(window=[3, 3], inp_ch_modes=np.array([4, 4, 4, 4]), out_ch_modes=np.array([8, 4, 4, 4]),
                   ranks=np.array([1, 9, 9, 9, 1]), strides=[1, 1], padding='same'),
            nn.ReLU(inplace=True),
            TTConv(window=[3, 3], inp_ch_modes=np.array([8, 4, 4, 4]), out_ch_modes=np.array([8, 4, 4, 4]),
                   ranks=np.array([1, 9, 9, 9, 1]), strides=[1, 1], padding='same'),
            nn.ReLU(inplace=True),
            TTConv(window=[3, 3], inp_ch_modes=np.array([8, 4, 4, 4]), out_ch_modes=np.array([8, 4, 4, 4]),
                   ranks=np.array([1, 9, 9, 9, 1]), strides=[1, 1], padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            TTConv(window=[3, 3], inp_ch_modes=np.array([8, 4, 4, 4]), out_ch_modes=np.array([8, 4, 4, 4]),
                   ranks=np.array([1, 9, 9, 9, 1]), strides=[1, 1], padding='same'),
            nn.ReLU(inplace=True),
            TTConv(window=[3, 3], inp_ch_modes=np.array([8, 4, 4, 4]), out_ch_modes=np.array([8, 4, 4, 4]),
                   ranks=np.array([1, 9, 9, 9, 1]), strides=[1, 1], padding='same'),
            nn.ReLU(inplace=True),
            TTConv(window=[3, 3], inp_ch_modes=np.array([8, 4, 4, 4]), out_ch_modes=np.array([8, 4, 4, 4]),
                   ranks=np.array([1, 9, 9, 9, 1]), strides=[1, 1], padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

