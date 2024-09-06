import torch.nn.functional as F
import torch.nn as nn
import math


from modules.white_dense_block import WhiteDenseBlock, WhiteTransition
from modules.white_bn import WhiteBatchNorm2d
from modules.white_conv import WhiteConv2
from modules.white_pooling import WhitePool
from modules.white_linear import WhiteLinear
from modules.white_net import WhiteNet
from modules.white_sequential import WhiteSequential



class DenseNet121(WhiteNet):
    def __init__(self, device=None):
        super(DenseNet121, self).__init__()
        self.device = device

        self.growth_rate = 12
        nblocks = [6, 12, 24, 16]
        block = WhiteDenseBlock
        reduction = 0.5
        num_classes = 10

        num_planes = 2*self.growth_rate
        self.conv1 = WhiteConv2(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*self.growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = WhiteTransition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*self.growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = WhiteTransition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*self.growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = WhiteTransition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*self.growth_rate

        self.bn = WhiteBatchNorm2d(num_planes, activate_function=F.relu)
        self.pool = WhitePool(num_planes, 4)
        self.linear = WhiteLinear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return WhiteSequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = self.bn(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out