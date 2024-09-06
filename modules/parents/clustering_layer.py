from enum import Enum

import numpy as np
import torch

from utils.codebook import encode
from utils.out_clustering import Net_Info


class Forward_Mode(Enum):
    normal = 0
    white = 1


class ClusteringLayer:

    def init_info(self, layer_name, previous):
        self.info = Net_Info(layer_name, previous)
        return [self]

    def get_info(self):
        return self.info

    def load_table_compressed(self):
        try:
            path,layer_name = Net_Info.save_path, self.info.layer_name
            white_table = np.load(path + "/" + layer_name + "_table.npz", allow_pickle=True)
            self.white_table = white_table['arr1']
            self.last_codebook = np.load(path + "/" + layer_name + "_last_codebook.npy", allow_pickle=True)
            return True
        except IOError:
            return False

    def load_table(self):
        try:
            path,layer_name = Net_Info.save_path, self.info.layer_name
            white_table = np.load(path + "/" + layer_name + "_table.npy", allow_pickle=True)
            self.white_table = white_table
            self.last_codebook = np.load(path + "/" + layer_name + "_last_codebook.npy", allow_pickle=True)
            return True
        except IOError:
            return False

    def load_last_codebook(self):
        try:
            path, layer_name = Net_Info.save_path, self.info.layer_name
            self.last_codebook = np.load(path + "/" + layer_name + "_last_codebook.npy", allow_pickle=True)
            return True
        except IOError:
            return False

    def nearest_value(self, value: float, cluster: np.array):
        return (np.abs(cluster - value)).argmin()

    def set_forward_mode(self, forward_mode:Forward_Mode):
        self.forward_mode = forward_mode

    def set_quantization_count(self, quantization_count):
        self.quantization_count = quantization_count

    def set_previous_codebook(self, previous_codebook):
        self.previous_codebook = previous_codebook

    def process_input(self, input): #输入有两种情况，正常输入和白盒输入，对这两种情况做转化。
        def process_single_input(input, previous):
            normal_input_set = [Forward_Mode.normal]
            white_input_set = [Forward_Mode.white]
            if self.forward_mode in normal_input_set and previous.forward_mode in white_input_set:
                input = previous.decode(input)
            if self.forward_mode in white_input_set and previous.forward_mode in normal_input_set:
                input = encode(input, previous.last_codebook) # 输入编码，正常输入变为白盒输入
            return input

        if isinstance(self.info.previous, tuple):
            left, right = input
            left = process_single_input(left, self.info.previous[0])
            right = process_single_input(right, self.info.previous[1])
            input = (left, right)
        else:
            input = process_single_input(input, self.info.previous)
        return input

    def forward(self, input):
        input = self.process_input(input)
        if self.forward_mode == Forward_Mode.normal:
            out = self._normal_forward(input)
        elif self.forward_mode == Forward_Mode.white:
            out = self._whitebox_forward(input)
        else:
            out = self._normal_forward(input)
        return out

    def _whitebox_forward(self, input) -> torch.Tensor:
        raise Exception("not implement")
