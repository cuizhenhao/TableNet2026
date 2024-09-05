import datetime

import torch
from torchvision import transforms
from torchvision.datasets import mnist, CIFAR10, CIFAR100, ImageFolder

from modules.parents.clustering_layer import Forward_Mode
from net.cct_imdb import CCT_IMDB
from net.lstm_imdb import NEW_LSTM_IMDB
from net.mobilenet import MobileNetV1
from net.resnet50 import WhiteResNet50
from utils.redirect_std import ReDirectSTD
import numpy as np

from net.densenet121 import DenseNet121
from net.lenet5 import LeNet5
from net.resnet20 import ResNet20

np.object = object

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--net', type=str, default='LeNet5',
                        help='CNN/LeNet5/ResNet50/BNLeNet5')
    parser.add_argument('--bit', type=int, default=4,
                        help='quantization count')
    parser.add_argument('--prune', action='store_true',
                        help='use prune model or not')
    parser.add_argument('--clustering_size', type=int, default=500,
                        help='size of clustering set')
    parser.add_argument('--model_path', type=str, default='result/prune/LeNet5.pt',
                        help='加载的模型参数的路径，比如result/prune/LeNet5.pt')
    args = parser.parse_args()
    print(args)
    print("start training")
    root = "./result/model"
    log_root = "./result/logs/table/" + str(args.bit)
    save_root = "./result/table"
    if args.prune:
        root = "./result/prune"
        log_root = "./result/logs/prune_table/" + str(args.bit)
        save_root = "./result/prune_table"

    os.makedirs(root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)

    white_bit = int(args.bit)
    clustering_batch_size = int(args.clustering_size)

    model_path = os.path.join(args.model_path)
    save_path = os.path.join(save_root, args.net + "/" + str(args.bit) + "/" + str(args.clustering_size))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    today = datetime.datetime.today()
    prefix = str(today.month) + str(today.day) + str(today.hour)
    stdout_file = os.path.join(
        log_root, args.net + prefix + ".log")
    stderr_file = os.path.join(
        log_root, args.net + prefix + "_error.log")
    ReDirectSTD(stdout_file, 'stdout | stderr', False)

    if args.net == "LeNet5":
        net = LeNet5()
    elif args.net == "ResNet20":
        net = ResNet20()
    elif args.net == "DenseNet121":
        net = DenseNet121()
    elif args.net == "MobileNetV1":
        net = MobileNetV1()
    elif args.net == "ResNet50-sp98":
        net = WhiteResNet50()
    elif args.net == "lstm-imdb":
        net = NEW_LSTM_IMDB()
    elif args.net == "CCT_IMDB":
        net = CCT_IMDB(vocab_size=10001, num_classes=1, kernel_size=1)
    print(net)

    if args.net == "LeNet5":
        mean = 0.5
        std = 0.5
        net.last_codebook = np.array([(256, (np.arange(256, dtype=np.float32) / 255 - mean) / std)], dtype=np.object)
        # 这里定义了反标准化的codebook
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x * 255)
            transforms.Normalize(0.5, 0.5)
        ])
        test_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        train_set = mnist.MNIST('./data', train=True, transform=train_tf, download=True)
        test_set = mnist.MNIST('./data', train=False, transform=test_tf, download=True)
        test_batch_size = 128
        clustering_batch_size = 500
        input_format = Forward_Mode.normal
    elif args.net == "ResNet20" or args.net == "DenseNet121":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        # net.last_codebook = np.array([
        #     (256, (np.arange(256, dtype=np.float) / 255 - mean[0]) / std[0]),
        #     (256, (np.arange(256, dtype=np.float) / 255 - mean[1]) / std[1]),
        #     (256, (np.arange(256, dtype=np.float) / 255 - mean[2]) / std[2])], dtype=np.object)
        net.last_codebook = np.array([(256, np.arange(256)), (256, np.arange(256)), (256, np.arange(256))], dtype=np.object)

        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x * 255),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x * 255),
        ])
        train_set = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_set = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_batch_size = 100
        clustering_batch_size = 500
        input_format = Forward_Mode.normal
    elif args.net == "ResNet50-sp98" or args.net == "MobileNetV1":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        net.last_codebook = np.array([
            (256, (np.arange(256, dtype=np.float32) / 255 - mean[0]) / std[0]),
            (256, (np.arange(256, dtype=np.float32) / 255 - mean[1]) / std[1]),
            (256, (np.arange(256, dtype=np.float32) / 255 - mean[2]) / std[2])], dtype=np.object)
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])

        train_set = ImageFolder(root='./data/imagenet/train-5000', transform=transform_train)
        test_set = ImageFolder(root='./data/imagenet/mintest', transform=transform_test)
        test_batch_size = 10
        clustering_batch_size = 5000
        input_format = Forward_Mode.normal
    elif args.net == "lstm-imdb":
        train_set = torch.load('./data/IMDB/train_data.pt')
        test_set = torch.load('./data/IMDB/valid_data.pt')
        test_batch_size = 64
        input_format = Forward_Mode.normal
        pass
    elif args.net == "CCT_IMDB":
        train_set = torch.load('./data/CCT_IMDB/train_data.pt')
        test_set = torch.load('./data/CCT_IMDB/valid_data.pt')
        test_batch_size = 100
        input_format = Forward_Mode.normal
        pass

    net.test_whitebox(save_path, train_set, test_set, test_batch_size, clustering_batch_size, white_bit, input_format)