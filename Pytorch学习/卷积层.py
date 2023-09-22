import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn

writer = SummaryWriter('logs')
data_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                        download=True)
data_loader = DataLoader(dataset=data_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)


# 建立神经网络
class TestNeuralNetwork(nn.Module):
    def __init__(self):
        super(TestNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


test_nn = TestNeuralNetwork()
for data in data_loader:
    imgs, targets = data
    # writer.add_images("input_imgs", imgs, 1, dataformats='NCHW')
    outputs = test_nn(imgs)
    # writer.add_images("output_imgs", outputs, 1, dataformats='NCHW')
    writer.add_image("input_imgs", imgs[0], 1, dataformats='CHW')
    # print(imgs)
    writer.add_image("output_imgs", outputs[0], 1, dataformats='CHW')
    # 一定要加上close
    writer.close()
    print(outputs)
    break
