import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, channel=256):
        super(Attention, self).__init__()
        self.fc1 = nn.Conv2d(channel, channel, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel, channel, 1, bias=False)
        self.att = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.fc2(self.relu1(self.fc1(x)))

        x = torch.sum(x, dim=1)
        x = x.reshape(b, 1, h, w)
        # print(x.shape)
        # x = self.att(x)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=256, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # print(self.max_pool(x).shape)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # print(max_out.shape)
        out = avg_out + max_out
        # print(avg_out.shape,max_out.shape)
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)