import numpy as np
import torch
import torchvision
from torch import nn


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, bn=True, activation=True, bias=True):
    op = [
            torch.nn.Conv2d(channels_in, channels_out,
                            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
    ]
    if bn:
        op.append(torch.nn.BatchNorm2d(channels_out))
    if activation:
        op.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*op)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class fast_resnet(nn.Module):
    
    def __init__(self, num_class=10, sphere_projected=False, last_relu=True, bias=False):
        super(fast_resnet, self).__init__()
        self.encoder = torch.nn.Sequential(
            conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
            conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
            # torch.nn.MaxPool2d(2),

            Residual(torch.nn.Sequential(
                conv_bn(128, 128),
                conv_bn(128, 128),
            )),

            conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2),

            Residual(torch.nn.Sequential(                             # try from here
                conv_bn(256, 256),
                conv_bn(256, 256),
            )),

            conv_bn(256, 128, kernel_size=3, stride=1, padding=0, activation=last_relu),

            torch.nn.AdaptiveMaxPool2d((1, 1)),
            Flatten())
        self.fc = torch.nn.Linear(128, num_class, bias=bias)
        self.sphere = sphere_projected
    
    def forward(self, x):
        x = self.encoder(x)
        if self.sphere:
            norms = torch.sqrt(torch.sum(x**2, 1)).unsqueeze(1).repeat(1, x.size()[1])
            x = x / norms
        x = self.fc(x)
        return (x)
