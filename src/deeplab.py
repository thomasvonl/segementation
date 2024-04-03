import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, depth: int, in_channel: int, out_channel: int, padding: int = 1, dilation: int = 1,
                 pool_stride: int = 2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=padding,
                      dilation=dilation),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=padding,
                      dilation=dilation),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        if depth == 3:
            self.conv_block.append(
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=padding,
                          dilation=dilation))
            self.conv_block.append(nn.BatchNorm2d(out_channel))
            self.conv_block.append(nn.ReLU(inplace=True))

        self.conv_block.append(nn.MaxPool2d(kernel_size=3, stride=pool_stride, padding=1))

    def forward(self, x):
        return self.conv_block(x)


class DeeplabV1(nn.Module):

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.n_channels = in_channels
        channels = [in_channels, 64, 128, 256, 512, 512]
        self.block1 = ConvBlock(depth=2, in_channel=channels[0], out_channel=channels[1])
        self.block2 = ConvBlock(depth=2, in_channel=channels[1], out_channel=channels[2])
        self.block3 = ConvBlock(depth=3, in_channel=channels[2], out_channel=channels[3])
        self.block4 = ConvBlock(depth=3, in_channel=channels[3], out_channel=channels[4], pool_stride=1)
        self.block5 = ConvBlock(depth=3, in_channel=channels[4], out_channel=channels[5], dilation=2, padding=2,
                                pool_stride=1)

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=12, dilation=12)
        self.normalization1 = nn.BatchNorm2d(1024)
        self.relu6 = nn.ReLU(inplace=True)
        # self.drop6 = nn.Dropout2d(0.5)

        self.conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, padding=0)
        self.normalization2 = nn.BatchNorm2d(1024)
        self.relu7 = nn.ReLU(inplace=True)
        # self.drop7 = nn.Dropout2d(0.5)

        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avg_pool(x)

        x = self.conv6(x)
        x = self.normalization1(x)
        x = self.relu6(x)
        # x = self.drop6(x)

        x = self.conv7(x)
        x = self.normalization2(x)
        x = self.relu7(x)
        # x = self.drop7(x)

        x = self.conv8(x)

        return x

# if __name__ == "__main__":
#     dv = DeeplabV1()
#     dummy_input = torch.rand(2, 1, 512, 512)
#     x = dv(dummy_input)
#     assert x.shape == (2, 1, 512, 512)
