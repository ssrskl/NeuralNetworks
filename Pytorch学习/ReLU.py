import torch
import torchvision
from torch import nn
from torch.nn import ReLU,Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')
data_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
data_loader = DataLoader(dataset=data_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)


class ReLUNeuralNetwork(nn.Module):
    def __init__(self):
        super(ReLUNeuralNetwork, self).__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        return x


reLUNeuralNetwork = ReLUNeuralNetwork()
for data in data_loader:
    imgs, targets = data
    writer.add_images("input_imgs_relu", imgs, 1, dataformats='NCHW')
    outputs = reLUNeuralNetwork(imgs)
    writer.add_images("output_imgs_relu", outputs, 1, dataformats='NCHW')
    writer.close()
    print(outputs)
    break
