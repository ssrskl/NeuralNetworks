# 训练CIFAR 10数据集
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss, Module, Sequential, Conv2d, MaxPool2d, Flatten, Linear

# 设置使用设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用的设备为：{}".format(device))  # 打印出当前使用的设备
# 准备数据集
train_dataset = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)
# 检测数据集的长度
print("训练集的长度为：{}".format(len(train_dataset)))
print("测试集的长度为：{}".format(len(test_dataset)))
# 加载数据集
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)


# 搭建神经网络
class NeuralNetwork(Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.module1 = torch.nn.Sequential(
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


# 创建神经网络
neural_network = NeuralNetwork().to(device)
# 创建损失函数
loss_function = CrossEntropyLoss()
loss_function = loss_function.cuda()
# 创建优化器
optimizer = torch.optim.SGD(neural_network.parameters(), lr=0.01)
# 设置训练的一些参数
epochs = 30
total_train_step = 0
total_test_step = 0
# 开始训练
for i in range(epochs):
    print("第{}个epoch开始训练".format(i))
    # 进入训练模式
    neural_network.train()
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = neural_network(imgs)
        loss_value = loss_function(outputs, targets)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("第{}个epoch的第{}个step的loss值为：{}".format(i, total_train_step, loss_value))
    # 每个epoch结束后，测试一下准确率
    total_test_loss = 0
    total_accuracy = 0
    # 进入测试模式
    neural_network.eval()
    with torch.no_grad():  # 测试的时候不需要计算梯度
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = neural_network(imgs)
            loss_value = loss_function(outputs, targets)
            total_test_loss += loss_value
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
            total_test_step += 1
        print("整体测试集的loss值为：{}".format(total_test_loss))
        print("整体测试集的准确率为：{}".format(total_accuracy / len(test_dataset)))

# 保存模型
torch.save(neural_network, "cifar10_30.pth")
