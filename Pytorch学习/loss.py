import torch
import torchvision
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, Sequential, Linear, Flatten, MaxPool2d, Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
data_loader = DataLoader(dataset=data_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

class LossNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNeuralNetwork, self).__init__()
        self.module1 = Sequential(
            Conv2d(3, 32, 5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, 5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, 5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.module1(x)
        return x

loss = CrossEntropyLoss()
lossNeuralNetwork = LossNeuralNetwork()
for data in data_loader:
    imgs, targets = data
    outputs = lossNeuralNetwork(imgs)
    loss_value = loss(outputs, targets)
    loss_value.backward()
    print(loss_value)
    break