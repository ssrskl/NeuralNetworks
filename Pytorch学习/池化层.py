import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# print(input.shape)
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)


writer = SummaryWriter('logs')
data_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
data_loader = DataLoader(dataset=data_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)


class PoolNeuralNetwork(nn.Module):
    def __init__(self):
        super(PoolNeuralNetwork, self).__init__()
        self.pool = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        x = self.pool(x)
        return x


poolNeuralNetwork = PoolNeuralNetwork()
for data in data_loader:
    imgs, targets = data
    writer.add_images("input_imgs", imgs, 0)
    print(imgs.shape)
    outputs = poolNeuralNetwork(imgs)
    print(outputs.shape)
    writer.add_images("output_imgs", outputs, 0, dataformats='NCHW')
    writer.close()
    break
