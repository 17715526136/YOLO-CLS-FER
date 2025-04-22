import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p=None, d=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, autopad(k, p, d))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class inception(nn.Module):
    def __init__(self, in_channels, out_channels, map_reduce=4):
        super(inception, self).__init__()
        self.out_channels = out_channels
        inter_planes = in_channels // map_reduce

        self.one = Conv1(in_channels,  inter_planes, 1, 1)
        self.one4 = Conv1(in_channels, inter_planes, 1, 2)

        self.one1 = nn.Sequential(
            Conv1(inter_planes, inter_planes, 3, 2),
        )
        self.one2 = nn.Sequential(
            Conv1(inter_planes, inter_planes, 3, 1),
            Conv1(inter_planes, inter_planes, 3, 2),
        )
        self.one3 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.one31 = Conv1(in_channels, inter_planes, 1, 1)
        # self.ConvLinear = Conv1(6 * inter_planes, out_channels, 1, 1)

    def forward(self, x):
        # 1x1短路连接
        # 1x1＋3x3
        # 1x1＋5x5
        # MAX3x3+1×1
        branch1x1 = self.one4(x)
        branch3x3 = self.one1(self.one(x))
        branch5x5 = self.one2(self.one(x))
        branch1x1_3x3 = self.one31(self.one3(x))
        x = torch.cat((branch1x1, branch3x3, branch5x5, branch1x1_3x3), 1)

        return x

class AlphaController(nn.Module):
    def __init__(self, input_dim):
        super(AlphaController, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()  # 将输出限制在0到1之间
        )
    def forward(self, x):
        return self.fc(x)


class DCMConv(nn.Module):
    def __init__(self, in_channels):
        super(DCMConv, self).__init__()
        self.controller = AlphaController(128)
        self.in_channels = in_channels

    def forward(self, x):
        # 使用控制器网络生成 alpha，并限制其在0到1之间
        alpha = 0.5

        # 计算通道划分
        in_channels_one = int(self.in_channels * alpha)
        in_channels_one = (in_channels_one + 3) // 4 * 4
        in_channels_two = self.in_channels - in_channels_one

        # 通道划分与处理
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 1, False, True)
        x1, x2 = torch.split(x, [in_channels_one, in_channels_two], dim=1)

        # 根据动态通道数定义卷积层
        one_conv = inception(in_channels_one, in_channels_one)
        two_conv = nn.Conv2d(in_channels_two, in_channels_two, 3, 2)

        # 通过动态卷积层进行处理
        x1 = one_conv(x1)
        x2 = two_conv(x2)

        return torch.cat((x1, x2), 1)

if __name__ == '__main__':
    net = DCMConv(64)
    from pytorch_model_summary import summary
    print(summary(net, torch.zeros(5, 3, 50, 50)))
    output_tensor = net(torch.zeros(5, 64, 50, 50))
    print("Input size:", (5, 64, 50, 50))
    print("Output size:", output_tensor.size())