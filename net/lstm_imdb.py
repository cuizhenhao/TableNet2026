import math
import pickle
import time

import torch.nn.functional as F

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader

from modules.cct.white_activation import WhiteActivation
from modules.cct.white_cat2d import WhiteCat2D
from modules.lstm.white_embedding import WhiteEmbedding
from modules.lstm.white_mul import WhiteMul
from modules.parents.clustering_layer import Forward_Mode, ClusteringLayer
from modules.white_dense_block import WhiteCat
from modules.white_linear import WhiteLinear
from modules.white_net import WhiteNet
from modules.white_res_block import WhiteAdd
from utils.codebook import encode, nearest_value
from utils.out_clustering import Net_Info


# function to create train, test data given stock data and sequence length
def load_data():
    # load data
    train_data = torch.load('./data/IMDB/train_data.pt')
    valid_data = torch.load('./data/IMDB/valid_data.pt')
    vocab = pickle.load(open('./data/IMDB/vocab.pkl', 'rb'))
    return [train_data, valid_data, vocab]


class CustomLSTM(WhiteNet):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.last_codebook = np.zeros(input_size+hidden_size, dtype=object)
        self.add_last_codebook = np.zeros(hidden_size, dtype=object)

        for i in range(input_size+hidden_size):
            self.last_codebook[i] = (256, torch.linspace(-1, 1, 256).numpy())
        for i in range(hidden_size):
            self.add_last_codebook[i] = (256, torch.linspace(-2, 2, 256).numpy())
        # self.cat = WhiteCat(input_size, hidden_size)
        self.W_i = WhiteLinear(input_size + hidden_size, hidden_size, activate_function=F.sigmoid, set_codebook=self.last_codebook[0])
        self.W_c = WhiteLinear(input_size + hidden_size, hidden_size, activate_function=F.tanh, set_codebook=self.last_codebook[0])
        self.W_f = WhiteLinear(input_size + hidden_size, hidden_size,activate_function=F.sigmoid, set_codebook=self.last_codebook[0])
        self.W_o = WhiteLinear(input_size + hidden_size, hidden_size, activate_function=F.sigmoid, set_codebook=self.last_codebook[0])

        self.mul1 = WhiteMul(hidden_size, set_codebook=self.last_codebook[0])
        self.mul2 = WhiteMul(hidden_size, set_codebook=self.last_codebook[0])
        self.add = WhiteAdd(hidden_size, set_codebook=self.add_last_codebook[0])
        self.mul3 = WhiteMul(hidden_size,  set_codebook=self.last_codebook[0])
        self.tanh = WhiteActivation(hidden_size, activate_function=F.tanh,  set_codebook=self.last_codebook[0])


    def forward(self, x, init_states=None):
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device)
            )
        else:
            h_t, c_t = init_states
        for t in range(seq_sz):
            x_t = x[:, t, :]
            combined = torch.cat((x_t, h_t), dim=-1)  # 合并输入x和隐藏状态h
            # combined = self.cat((x_t, h_t))
            # print(combined.min(), combined.max())

            i_t = self.W_i(combined)
            g_t = self.W_c(combined)
            f_t = self.W_f(combined)
            o_t = self.W_o(combined)

            # c_t = f_t * c_t + i_t * g_t
            c_t2 = self.mul2((i_t, g_t))
            c_t1 = self.mul1((f_t, c_t))
            c_t = self.add((c_t1, c_t2))

            tanh = self.tanh(c_t)
            h_t = self.mul3((o_t, tanh))
            # print(c_t.min(), c_t.max())
            # print(h_t.min(), h_t.max())

            # h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

    def init_info(self, layer_name, previous):
        network = []
        # network.extend(self.cat.init_info(layer_name + ".cat",previous = previous))
        self.add.last_codebook = self.add_last_codebook
        self.mul3.last_codebook = self.last_codebook
        network.extend(self.W_i.init_info(layer_name + ".W_i",previous = self.mul3))
        network.extend(self.W_c.init_info(layer_name + ".W_c",previous = self.mul3))

        network.extend(self.W_f.init_info(layer_name + ".W_f",previous = self.mul3))
        network.extend(self.W_o.init_info(layer_name + ".W_o",previous = self.mul3))
        network.extend(self.mul2.init_info(layer_name + ".mul2",previous = (self.W_i, self.W_c)))

        network.extend(self.mul1.init_info(layer_name + ".mul1",previous = (self.W_f, self.add)))
        network.extend(self.add.init_info(layer_name + ".add",previous = (self.mul1, self.mul2)))
        network.extend(self.tanh.init_info(layer_name + ".tanh",previous =  self.add))
        network.extend(self.mul3.init_info(layer_name + ".mul3",previous = (self.W_o, self.tanh)))

        return network


# function to predict accuracy
def acc(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

# Here we define our model as a class
class LSTM_IMDB(WhiteNet):
    def __init__(self, input_dim, hidden_dim, output_dim, vocab_size):
        super(LSTM_IMDB, self).__init__()
        self.hidden_dim = hidden_dim
        self.last_codebook = np.zeros(hidden_dim+input_dim, dtype=object)
        for i in range(hidden_dim+input_dim):
            self.last_codebook[i] = (256, torch.linspace(-1, 1, 256).numpy())

        self.embedding = WhiteEmbedding(vocab_size, 500, input_dim, set_codebook=self.last_codebook[0])
        self.lstm = CustomLSTM(input_dim, hidden_dim)
        self.fc = WhiteLinear(hidden_dim, output_dim, bias=True, is_last_layer=True)

    def forward(self, x, hidden):
        batch_size = x.size(0)
         # shape: B x S x Feature   since batch = True
        self.embedding.info.previous.forward_mode = self.embedding.forward_mode
        embeds = self.embedding(x)
        h0 = torch.zeros(x.size(0), self.hidden_dim) + 128
        c0 = torch.zeros(x.size(0), self.hidden_dim) + 128
        if self.lstm.mul3.forward_mode != Forward_Mode.white:
            embeds = self.embedding._normal_forward(x)
            h0 = torch.zeros(x.size(0), self.hidden_dim)
            c0 = torch.zeros(x.size(0), self.hidden_dim)

        hidden = (h0, c0)
        # print(embeds.max(), embeds.min())
        lstm_out, hidden  = self.lstm(embeds, hidden)
        lstm_out = lstm_out[:, -1, :]
        print(lstm_out.max(), lstm_out.min())

        out = self.fc(lstm_out).squeeze(0)
        out = torch.sigmoid(out)
        return out, hidden

    def init_network(self, input_format=Forward_Mode.normal):
        torch.multiprocessing.set_sharing_strategy('file_system')

        head = ClusteringLayer()
        head.last_codebook = self.last_codebook
        head.forward_mode = input_format
        network = []
        previous = head
        network.extend(self.embedding.init_info("embedding",previous = previous))
        network.extend(self.lstm.init_info("lstm",previous = previous))
        network.extend(self.fc.init_info( "fc",previous = network[-1]))

        for module in network:
            print(module, module.info)
        return network


    def encode_input(self, x_test):
        encode_input = x_test.clone()
        for in_ch in range(encode_input.shape[1]):
            origin_shape = x_test[:, in_ch].shape
            on_ch = encode_input[:, in_ch].reshape(-1)
            codebook_tensor = torch.from_numpy(self.last_codebook[0][1])
            abs_diff = torch.abs(on_ch[:, None] - codebook_tensor)
            nearest_indices = torch.argmin(abs_diff, dim=1)
            on_ch[:] = nearest_indices
            encode_input[:, in_ch] = on_ch.reshape(origin_shape)
        return encode_input.to(torch.uint8)

    def start_whitebox(self, save_path, train_set, test_set, test_batch_size, clustering_batch_size, white_bit, input_format):
        train_data, valid_data, vocab = load_data()
        Net_Info.set_save_path(save_path)
        network = self.init_network(input_format)
        # network = self.init_network(Forward_Mode.normal)
        clustering_batch_size = 1000
        train_loader = DataLoader(train_data, shuffle=True, batch_size=clustering_batch_size)
        inputs, labels = train_loader.__iter__().__next__()

        self.eval()
        for module in network:
            module.set_previous_codebook(module.info.get_previous_codebook())
            if module.load_table():
                module.set_forward_mode(Forward_Mode.white)
                print("="*30 + module.info.layer_name + " load" + "="*30)
            else:
                module.set_quantization_count(1 << white_bit)
                module.set_previous_codebook(module.info.get_previous_codebook())
                module.set_forward_mode(Forward_Mode.cluster_multiprocessing)
                print("="*30 + module.info.layer_name + " clustering" + "="*30)
                with torch.no_grad():
                    # x_train = self.encode_input(x_train)
                    output,h = self(inputs, None)
                    accuracy = acc(output, labels)
                    print('acc of the network after '+module.info.layer_name +' clustering: ', accuracy*100 / len(labels))
                print("="*30 + module.info.layer_name + " generate table" + "="*30)
                module.set_forward_mode(Forward_Mode.white)
                module.generate_table_multiprocessing()

    def test_whitebox(self, save_path, train_set, test_set, test_batch_size, clustering_batch_size, white_bit, input_format):
        train_data, valid_data, vocab = load_data()
        Net_Info.set_save_path(save_path)
        network = self.init_network(input_format)
        # network = self.init_network(Forward_Mode.normal)

        self.eval()
        start = time.time()
        for i, module in enumerate(network):
            module.set_previous_codebook(module.info.get_previous_codebook())
            if module.load_table_compressed():
                print("="*30 + module.info.get_name() + " load" + "="*30)
                module.set_forward_mode(Forward_Mode.white)
            else:
                # module.set_previous_codebook(module.info.get_previous_codebook())
                break
        end = time.time()
        print("load time: ", end - start)

        batch_size = 200
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)
        val_acc = 0.0
        val_h = None
        total = 0
        start = time.time()
        for i, (inputs, labels) in enumerate(valid_loader):
            val_h = tuple([each.data for each in val_h]) if val_h is not None else None

            output, val_h = self(inputs, val_h)
            accuracy = acc(output, labels)
            val_acc += accuracy
            total += len(labels)
            print("Validation Accuracy{}: {:.4f}".format(i, val_acc*100 /total))
        end = time.time()
        print("infer time: ", end - start)


def NEW_LSTM_IMDB():
    input_dim = 16
    hidden_dim = 32
    output_dim = 1
    model = LSTM_IMDB(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, vocab_size=1001)
    return model