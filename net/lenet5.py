import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modules.parents.clustering_layer import Forward_Mode
from modules.white_conv import WhiteConv2
from modules.white_pooling import WhitePool
from modules.white_linear import WhiteLinear
from modules.white_net import WhiteNet
from utils.out_clustering import Net_Info


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_fgsm(model, X, y, epsilon):
    delta = torch.zeros_like(X, requires_grad=True)
    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    loss.backward()
    grad = delta.grad.detach()
    delta.data = epsilon * torch.sign(grad)
    return delta.detach()


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0])
    max_delta = torch.zeros_like(X)
    for _ in range(restarts):
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
        delta.data = clamp(delta, 0-X, 1-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)[0]
            if len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            d = clamp(d, 0-X, 1-X)
            delta.data[index] = d[index]
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


class LeNet5(WhiteNet):
    def __init__(self, device=None):
        super(LeNet5, self).__init__()
        self.device = device        
        self.mean = (0.1307,)
        self.std = (0.3081,)

        self.conv1 = WhiteConv2(1, 6, 5, padding=2, activate_function=F.relu)#, is_first=True, mean=self.mean, std=self.std)
        self.pool1 = WhitePool(6, 2)
        self.conv2 = WhiteConv2(6, 16, 5, activate_function=F.relu)
        self.pool2 = WhitePool(16, 2)
        self.fc1 = WhiteLinear(16 * 5 * 5, 120, activate_function=F.relu)  # 6*6 from image dimension
        self.fc2 = WhiteLinear(120, 84, activate_function=F.relu)
        self.fc3 = WhiteLinear(84, 10, is_last_layer=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def test_by_attack(self, table_path, test_set, attack):
        test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

        Net_Info.set_save_path(table_path)
        network = self.init_network()
        self.eval()
        for i, module in enumerate(network):
            module.set_previous_codebook(module.info.get_previous_codebook())
            if module.load_table():
                print("=" * 30 + str(module.info) + " load" + "=" * 30)
                module.set_forward_mode(Forward_Mode.white)
            else:
                break

        self.eval()
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if self.device is not None:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            if attack == 'none':
                with torch.no_grad():
                    outputs = self(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += (outputs.max(1)[1] == targets).sum().item()

                    print(batch_idx, len(test_loader), 'Acc: %.3f%%'
                          % (100. * correct / total))
            else:
                model = LeNet5()
                model.init_network()
                model.eval()
                if attack == 'fgsm':
                    state_dict = torch.load('adv_training/LeNet5/eval/6-no.pt')
                    model.load_state_dict(state_dict, strict=False)
                    delta = attack_fgsm(model, inputs, targets, 0.1)
                elif attack == 'pgd':
                    state_dict = torch.load('adv_training/LeNet5/eval/4-fgsm.pt')
                    model.load_state_dict(state_dict, strict=False)
                    delta = attack_pgd(model, inputs, targets, 0.1, 1e-2, 50, 10)
                with torch.no_grad():
                    outputs = self(inputs + delta)
                    total += targets.size(0)
                    correct += (outputs.max(1)[1] == targets).sum().item()

                    print(batch_idx, len(test_loader), 'Acc: %.3f%%'
                          % (100. * correct / total))
