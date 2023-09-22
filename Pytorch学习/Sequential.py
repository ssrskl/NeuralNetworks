import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

data_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)


class MySequential(nn.Module):
    def __init__(self):
        super(MySequential, self).__init__()
        self.module1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.module1(x)
        return x


writer = SummaryWriter('logs')
for data in data_loader:
    imgs, targets = data
    my_sequential = MySequential()
    outputs = my_sequential(imgs)
    writer.add_graph(my_sequential, imgs)
    writer.add_images("input_imgs", imgs, 1, dataformats='NCHW')
    writer.close()
    print(outputs)
    break
