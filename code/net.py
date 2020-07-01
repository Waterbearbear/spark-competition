import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),  # 添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class unet(nn.Module):
    def __init__(self, input_channel=1, input_size=256, output_channel=11):
        super(unet, self).__init__()
        self.conv1 = DoubleConv(input_channel, 64)
        self.pool1 = nn.MaxPool2d(2)  # 128,128
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)  # 64,64
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)  # 32,32
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)  # 16,16
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)  # 32,32
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)  # 64,64
        self.conv8 = nn.Conv2d(256, 128, 1)
        self.conv9 = nn.Conv2d(128, output_channel, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        c8 = self.conv8(c7)
        c8 = nn.ReLU(inplace=True)(c8)
        c9 = self.conv9(c8)
        out = nn.Sigmoid()(c9)
        return out
