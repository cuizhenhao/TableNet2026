import random

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from modules.parents.clustering_layer import Forward_Mode, ClusteringLayer
from utils.out_clustering import Net_Info

class WhiteNet(nn.Module):
    def __init__(self, device=None):
        super(WhiteNet, self).__init__()
        self.device = device
        self.last_codebook = np.array([(256, np.arange(256)), (256, np.arange(256)), (256, np.arange(256))], dtype=object)
        self.setup_seed(20)

    def test_once(self, testloader, best_acc = None, save_path = None):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if self.device is not None:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self(inputs)
                if outputs.size()[1] == 1:
                    predicted = torch.round(nn.Sigmoid()(outputs).squeeze())
                else:
                    _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                print(batch_idx, len(testloader), 'Acc: %.3f%%'
                            % (100.*correct/total))
            if best_acc is not None and save_path is not None:
                acc = 100.*correct/total
                if acc > best_acc:
                    print("saving:", save_path)
                    torch.save(self.state_dict(), save_path+'-'+str(acc))
                    best_acc = acc
        return best_acc

    def init_info(self, layer_name, previous):
        network = []
        for _, (name, module) in enumerate(self.named_children()):
            sub_net = module.init_info(layer_name + "." + name, previous)
            network.extend(sub_net)
            previous = sub_net[-1]
        return network

    def init_network(self, input_format=Forward_Mode.normal):
        torch.multiprocessing.set_sharing_strategy('file_system')

        head = ClusteringLayer()
        head.last_codebook = self.last_codebook
        head.forward_mode = input_format
        network = []
        previous = head
        for (name, module) in self.named_children():
            sub_net = module.init_info(name, previous)
            network.extend(sub_net)
            previous = sub_net[-1]
        return network

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def test_whitebox(self, save_path, train_set, test_set, test_batch_size, clustering_batch_size, white_bit, input_format):
        test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)
        Net_Info.set_save_path(save_path)
        network = self.init_network(input_format)
        self.eval()
        for i, module in enumerate(network):
            module.set_previous_codebook(module.info.get_previous_codebook())
            if module.load_table_compressed():
                print("="*30 + module.info.get_name() + " load" + "="*30)
                module.set_forward_mode(Forward_Mode.white)
            else:
                break
        self.test_once(test_loader)
