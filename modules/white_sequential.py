from typing import Any

import torch.nn as nn
from .white_net import WhiteNet

class WhiteSequential(nn.Sequential, WhiteNet):
    def __init__(self, *args: Any):
        super(WhiteSequential, self).__init__(*args)
        self.previous_codebook = None

    def init_info(self, layer_name, previous):
        network = []
        for _, (name, module) in enumerate(self.named_children()):
            sub_net = module.init_info(layer_name + "." + name, previous)
            network.extend(sub_net)
            previous = sub_net[-1]
        return network

    def set_is_last_layer(self, is_last_layer):
        self[-1].set_is_last_layer(is_last_layer)

    def get_size(self) -> int:
        size = 0
        for _, (name, module) in enumerate(self.named_children()):
            size += module.get_size()
        return size