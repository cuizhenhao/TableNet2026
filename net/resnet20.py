import torch.nn.functional as F

from modules.white_res_block import WhiteResBlock
from modules.white_bn import WhiteBatchNorm2d
from modules.white_conv import WhiteConv2
from modules.white_pooling import WhitePool
from modules.white_linear import WhiteLinear
from modules.white_net import WhiteNet
from modules.white_sequential import WhiteSequential


class ResNet20(WhiteNet):
    def __init__(self, num_classes=10, device=None):
        super(ResNet20, self).__init__()
        self.in_planes = 16
        self.device = device
        block = WhiteResBlock

        self.conv1 = WhiteConv2(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False, padding_mode="replicate")
        self.bn1 = WhiteBatchNorm2d(16, activate_function=F.relu)
        self.layer1 = self._make_layer(block, 16, 3, stride=1)
        self.layer2 = self._make_layer(block, 32, 3, stride=2)
        self.layer3 = self._make_layer(block, 64, 3, stride=2)
        self.pool = WhitePool(64, 8)
        self.linear = WhiteLinear(64, num_classes, is_last_layer=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return WhiteSequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out