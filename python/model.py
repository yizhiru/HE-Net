import torch
import torch.nn as nn


class polynom_act(nn.Module):
    """二次激活 αx² + βx + c，便于 CKKS 求值。"""

    def __init__(self, alpha=None, beta=None, c=None):
        super(polynom_act, self).__init__()
        self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)
        self.beta = nn.Parameter(torch.randn(1), requires_grad=True)
        self.c = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        return (self.alpha * (x ** 2) + self.beta * x + self.c)


class CNNNet(nn.Module):
    """HE 友好的 4 段卷积分类网络，默认 128 通道起步。"""
    def __init__(self, num_classes, filters=128):
        super(CNNNet, self).__init__()
        first_f = filters
        second_f = filters * 2
        third_f = filters * 4
        forth_f = filters * 1
        self.conv1 = nn.Conv2d(3, first_f, kernel_size=(3, 3), stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(first_f)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        self.act1 = polynom_act()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(first_f, second_f, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(second_f)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()
        self.act2 = polynom_act()
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(second_f, third_f, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(third_f)
        self.bn3.weight.data.fill_(1)
        self.bn3.bias.data.zero_()
        self.act3 = polynom_act()
        # self.avg_pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(third_f, forth_f, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.bn4 = nn.BatchNorm2d(forth_f)
        self.bn4.weight.data.fill_(1)
        self.bn4.bias.data.zero_()
        self.act4 = polynom_act()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(forth_f, num_classes)

    def forward(self, x):
        # B1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.avg_pool1(out)
        # B2
        out = self.conv2(out)
        out = self.act2(self.bn2(out))
        out = self.avg_pool2(out)
        # B3
        out = self.conv3(out)
        out = self.act3(self.bn3(out))
        # out = self.avg_pool3(out)
        # B4
        out = self.conv4(out)
        out = self.act4(self.bn4(out))
        # avg pool
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        # dense
        out = self.linear(out)
        return out
