from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

imagenet_transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

imagenet_transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])  

def load_imagenet_testloader():
    test_set = ImageFolder(root='./data/imagenet/mintest', transform=imagenet_transform_test)
    test_batch_size = 128
    testloader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    return testloader

def load_imagenet_trainloader():
    test_set = ImageFolder(root='./data/imagenet/mintest', transform=imagenet_transform_train)
    test_batch_size = 128
    trainloader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    return trainloader