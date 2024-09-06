

import torch.nn.functional as F
import torch.nn as nn

from modules.white_max_pooling import WhiteMaxPool
from modules.white_mini_linear import WhiteMiniLinear
from modules.white_res_block import WhiteResBlock, WhiteAdd
from modules.white_bn import WhiteBatchNorm2d
from modules.white_conv import WhiteConv2
from modules.white_pooling import WhitePool
from modules.white_linear import WhiteLinear
from modules.white_net import WhiteNet
from modules.white_sequential import WhiteSequential


class WhiteBottleneck(WhiteNet):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
    ) -> None:
        super(WhiteBottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = WhiteMiniConv2(inplanes, out_channels=width, kernel_size=1, bias=False)
        self.conv1 = WhiteConv2(inplanes, out_channels=width, kernel_size=1, bias=False)

        self.bn1 = WhiteBatchNorm2d(width,activate_function=F.relu)
        self.conv2 = WhiteConv2(width, width,3,stride= stride,padding=1,bias=False)
        # self.conv2 = WhiteMiniConv2(width, width,3,stride= stride,padding=1,bias=False)

        self.bn2 = WhiteBatchNorm2d(width,activate_function=F.relu)
        self.conv3 = WhiteConv2(width, planes * self.expansion, kernel_size=1,bias=False)
        # self.conv3 = WhiteMiniConv2(width, planes * self.expansion, kernel_size=1,bias=False)
        self.bn3 = WhiteBatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.add = WhiteAdd(planes * self.expansion, activate_function=F.relu)
        self.stride = stride

    def init_info(self, layer_name, previous):
        self.conv1.init_info(layer_name + ".conv1", previous=previous)
        self.bn1.init_info(layer_name + ".bn1",previous = self.conv1)
        self.conv2.init_info(layer_name + ".conv2",previous = self.bn1)
        self.bn2.init_info(layer_name + ".bn2",previous= self.conv2)
        self.conv3.init_info(layer_name + ".conv3",previous = self.bn2)
        self.bn3.init_info(layer_name + ".bn3",previous= self.conv3)

        left_previous, network = previous, [self.conv1,self.bn1,self.conv2,self.bn2, self.conv3,self.bn3]
        if self.downsample is not None:
            sub_net = self.downsample.init_info(layer_name + ".downsample",previous= previous)
            network.extend(sub_net)
            left_previous = sub_net[-1]

        self.add.init_info(layer_name + ".add",previous=(left_previous, self.bn3))
        network.append(self.add)
        return network

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.add((identity, out))

        return out


class WhiteResNet50(WhiteNet):
    def __init__(self, num_classes=1000, device=None):
        super(WhiteResNet50, self).__init__()
        self.inplanes = 64
        self.device = device
        block = WhiteBottleneck
        self.conv1 = WhiteConv2(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        # self.conv1 = WhiteMiniConv2(3, 64, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        self.bn1 = WhiteBatchNorm2d(64, activate_function=F.relu)
        self.maxpool = WhiteMaxPool(64, pool_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, 3, stride=1)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)
        self.layer3 = self._make_layer(block, 256, 6, stride=2)
        self.layer4 = self._make_layer(block, 512, 3, stride=2)

        self.avgpool = WhitePool(2048, 7)
        # self.avgpool = WhiteMaxPool(2048, 7)
        self.fc = WhiteMiniLinear(2048, num_classes, is_last_layer=True)
        # self.fc = WhiteLinear(2048, num_classes, is_last_layer=True)


    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = WhiteSequential(
                WhiteConv2(self.inplanes, planes * block.expansion,kernel_size=1,stride=stride,bias=False),
                # WhiteMiniConv2(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                WhiteBatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        return WhiteSequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if  isinstance(x, tuple):
            x1 = x[0].view(x[0].size(0), -1)
            x2 =x[1].view(x[1].size(0), -1)
            x = (x1,x2)
        else:
            x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x