import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def var_init(shape, initializer=None, regularizer=None, trainable=True, device='cpu'):
    var = torch.empty(shape, device=device)
    if initializer is not None:
        var = initializer(var)
    if regularizer is not None:
        var = regularizer(var)
    return nn.Parameter(var, requires_grad=trainable)


class TTConvFull(nn.Module):
    def __init__(self, 
                 window,
                 inp_ch_modes,              
                 out_ch_modes,
                 ranks,
                 strides=(1, 1),
                 padding='same',
                 filters_initializer=None,
                 filters_regularizer=None,
                 cores_initializer=None,
                 cores_regularizer=None,
                 biases_initializer=torch.zeros_,
                 biases_regularizer=None,
                 trainable=True,
                 device='cpu'):
        super(TTConvFull, self).__init__()

        self.window = window
        self.inp_ch_modes = inp_ch_modes
        self.out_ch_modes = out_ch_modes
        self.ranks = ranks
        self.strides = strides
        self.padding = padding.upper()
        self.filters_initializer = filters_initializer
        self.filters_regularizer = filters_regularizer
        self.cores_initializer = cores_initializer
        self.cores_regularizer = cores_regularizer
        self.biases_initializer = biases_initializer
        self.biases_regularizer = biases_regularizer
        self.trainable = trainable
        self.device = device

        filters_shape = [window[0], window[1], 1, ranks[0]]
        if window[0] * window[1] * 1 * ranks[0] == 1:
            self.filters = var_init(filters_shape, initializer=torch.ones_, trainable=False, device=device)
        else:
            self.filters = var_init(filters_shape, initializer=filters_initializer, regularizer=filters_regularizer, trainable=trainable, device=device)

        d = len(inp_ch_modes)
        self.cores = nn.ModuleList()
        for i in range(d):
            if isinstance(cores_initializer, list):
                cinit = cores_initializer[i]
            else:
                cinit = cores_initializer

            if isinstance(cores_regularizer, list):
                creg = cores_regularizer[i]
            else:
                creg = cores_regularizer

            core_shape = [out_ch_modes[i] * ranks[i + 1], ranks[i] * inp_ch_modes[i]]
            core = var_init(core_shape, initializer=cinit, regularizer=creg, trainable=trainable, device=device)
            self.cores.append(core)

        if biases_initializer is not None:
            self.biases = var_init([np.prod(out_ch_modes)], initializer=biases_initializer, regularizer=biases_regularizer, trainable=trainable, device=device)
        else:
            self.biases = None

    def forward(self, inp):
        inp_shape = inp.shape[1:]
        inp_h, inp_w, inp_ch = inp_shape[0:3]
        tmp = inp.view(-1, inp_h, inp_w, inp_ch)

        full = self.filters

        d = len(self.inp_ch_modes)
        for i in range(d):
            full = full.view(-1, self.ranks[i])
            core = self.cores[i].T
            core = core.view(self.ranks[i], -1)
            full = torch.matmul(full, core)

        out_ch = np.prod(self.out_ch_modes)

        fshape = [self.window[0], self.window[1]]
        order = [0, 1]
        inord = []
        outord = []
        for i in range(d):
            fshape.append(self.inp_ch_modes[i])
            inord.append(2 + 2 * i)
            fshape.append(self.out_ch_modes[i])
            outord.append(2 + 2 * i + 1)
        order += inord + outord
        full = full.view(fshape)
        full = full.permute(order)
        full = full.view(self.window[0], self.window[1], inp_ch, out_ch)

        padding = self.padding
        if padding == 'same':
            padding = (self.window[0] // 2, self.window[1] // 2)
        elif padding == 'valid':
            padding = 0

        tmp = F.conv2d(tmp, full, stride=self.strides, padding=padding)

        if self.biases is not None:
            out = tmp + self.biases
        else:
            out = tmp

        return out
