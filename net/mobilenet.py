from math import floor

import torch.nn.functional as F
from modules.white_mini_linear import WhiteMiniLinear
from modules.white_bn import WhiteBatchNorm2d
from modules.white_conv import WhiteConv2
from modules.white_conv_groups import WhiteConv2Groups
from modules.white_net import WhiteNet
from modules.white_pooling import WhitePool
from modules.white_sequential import WhiteSequential


class MobileNetV1(WhiteNet):
    def __init__(self):
        super(MobileNetV1, self).__init__()

        def depthwise_conv(n_ifm, n_ofm, stride):
            return WhiteSequential(
                WhiteConv2Groups(n_ifm, n_ifm, 3, stride=stride, padding=1, groups=n_ifm, bias=False, padding_mode='zeros'),
                WhiteBatchNorm2d(n_ifm, activate_function=F.relu),
                WhiteConv2(n_ifm, n_ofm, 1, stride=1, padding=0, bias=False, padding_mode='zeros'),
                WhiteBatchNorm2d(n_ofm, activate_function=F.relu)
            )

        base_channels = [32, 64, 128, 256, 512, 1024]
        self.channels = [max(floor(n * 1), 8) for n in base_channels]

        self.model = WhiteSequential(
            WhiteSequential(
                # *conv_bn_relu(3, self.channels[0], 3, stride=2, padding=1)),
                WhiteConv2(3, self.channels[0], 3, stride=2, padding=1, padding_mode='zeros'),
                WhiteBatchNorm2d(self.channels[0], activate_function=F.relu)),
            depthwise_conv(self.channels[0], self.channels[1], 1),
            depthwise_conv(self.channels[1], self.channels[2], 2),
            depthwise_conv(self.channels[2], self.channels[2], 1),
            depthwise_conv(self.channels[2], self.channels[3], 2),
            depthwise_conv(self.channels[3], self.channels[3], 1),
            depthwise_conv(self.channels[3], self.channels[4], 2),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[5], 2),
            depthwise_conv(self.channels[5], self.channels[5], 1),
            WhitePool(1024, 7),
        )
        self.fc = WhiteMiniLinear(self.channels[5], 1000, is_last_layer=True)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        return x
