# 测试自己的模型
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

img_path = './imgs/airplane.png'
image = Image.open(img_path)
image = image.convert('RGB')
# 定义图片的预处理
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.module1 = nn.Sequential(
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


# 导入模型
model = torch.load('./models/cifar10_30.pth',map_location=torch.device('cuda:0'))
image = torch.reshape(image, (1, 3, 32, 32))
image = image.cuda()
# 将模型设置为验证模式
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
# 输出最大值的索引
print(torch.argmax(output))