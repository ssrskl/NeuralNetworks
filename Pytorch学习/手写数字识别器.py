import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# 配置参数
writer = SummaryWriter('logs')
image_size = 28 * 28
num_classes = 10
num_epochs = 20
batch_size = 64
# 加载测试集
train_dataset = torchvision.datasets.MNIST(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                           download=True)
# 加载测试集
test_dataset = torchvision.datasets.MNIST(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 使用DataLoader加载数据集
train_data_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 定义卷积神经网络的参数
depth = [4, 8]


# 构建卷积神经网络
class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二层卷积
        self.conv2 = nn.Conv2d(depth[0], depth[1], kernel_size=5, stride=1, padding=2)
        # 建立一个线性连接层，输入尺寸为最后一层立方体的线性平铺，输出层512个节点
        self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1], 512)
        self.fc2 = nn.Linear(512, num_classes)

    # 定义前向传播
    def forward(self, x):
        # 第一层卷积和池化
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        # 第二层卷积
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)


for data in train_data_loader:
    images, targets = data
    print(images[0])
    # 显示第一张图片
    writer.add_image("MNIST", images[0], 1, dataformats='CHW')
    print(targets[0])
    break
