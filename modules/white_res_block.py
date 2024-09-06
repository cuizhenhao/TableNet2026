import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

from modules.parents.clustering_layer import ClusteringLayer, Forward_Mode
from .white_bn import WhiteBatchNorm2d
from .white_conv import WhiteConv2
from .white_net import WhiteNet

class WhiteResBlock(WhiteNet):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(WhiteResBlock, self).__init__()
        self.conv1 = WhiteConv2(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode="replicate")
        self.bn1 = WhiteBatchNorm2d(planes, activate_function=F.relu)
        self.conv2 = WhiteConv2(planes, planes, kernel_size=3, padding=1, bias=False, padding_mode="replicate")
        self.bn2 = WhiteBatchNorm2d(planes, activate_function=F.relu)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                WhiteConv2(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False, padding_mode="replicate"),
                WhiteBatchNorm2d(planes)
            )
        self.add = WhiteAdd(planes, activate_function=F.relu)
        self.previous_codebook = None

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        right = self.shortcut(x)
        out = self.add((out,right))
        return out

    def init_info(self, layer_name, previous):
        self.conv1.init_info(layer_name + ".conv1", previous=previous)
        self.bn1.init_info(layer_name + ".bn1",previous = self.conv1)
        self.conv2.init_info(layer_name + ".conv2",previous = self.bn1)
        self.bn2.init_info(layer_name + ".bn2",previous= self.conv2)
        right_previous, network = previous, [self.conv1,self.bn1,self.conv2,self.bn2]
        if len(self.shortcut) > 0:
            self.shortcut[0].init_info(layer_name + ".shortcut.0", previous=previous)
            self.shortcut[1].init_info(layer_name + ".shortcut.1", previous=self.shortcut[0])
            right_previous = self.shortcut[1]
            network.append(self.shortcut[0])
            network.append(self.shortcut[1])

        self.add.init_info(layer_name + ".add",previous=(self.bn2, right_previous))
        network.append(self.add)
        return network

    def set_is_last_layer(self, is_last_layer):
        self.add.set_is_last_layer(is_last_layer)

    def get_size(self) -> int:
        size = self.bn1.get_size() + self.conv1.get_size() + self.bn2.get_size() + self.conv2.get_size()
        if len(self.shortcut) > 0:
            size += self.shortcut[0].get_size() + self.shortcut[1].get_size()
        return size


class WhiteAdd(ClusteringLayer, nn.Module):
    def __init__(self, in_channels: int, activate_function = None, merge_last_codebook= False, set_codebook = None):
        super(WhiteAdd, self).__init__()
        self.white_table = np.zeros(in_channels, dtype=np.object)
        self.last_codebook = np.zeros(in_channels, dtype=np.object)
        self.codebook = np.zeros(in_channels, dtype=np.object)
        self.left_previous_codebook = None
        self.right_previous_codebook = None

        self.quantization_count = None
        self.activate_function = activate_function
        self.forward_mode = Forward_Mode.normal
        self.is_last_layer = False

        self.in_channels = in_channels
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)
        self.merge_last_codebook = merge_last_codebook
        self.set_codebook = set_codebook


    def set_is_last_layer(self, is_last_layer):
        self.is_last_layer = is_last_layer
        
    def _whitebox_forward(self, input) -> torch.Tensor:
        left, right = input
        left_arr = left.cpu().numpy().astype(np.uint8)
        right_arr = right.cpu().numpy().astype(np.uint8)
        if self.is_last_layer:
            new_outputs = np.zeros_like(left_arr, dtype=np.float32)
        for inch in range(self.in_channels):
            left_arr[:, inch] = self.white_table[inch][left_arr[:, inch], right_arr[:, inch]]
        
            if self.is_last_layer:
                new_outputs[:, inch] = self.last_codebook[inch][1][left_arr[:, inch]]
        if self.is_last_layer:
            return torch.Tensor(new_outputs)
        return torch.Tensor(left_arr)

    def _normal_forward(self, input) -> torch.Tensor:
        left, right = input
        out = left + right
        if self.activate_function is not None:
            out = self.activate_function(out)
        return out
