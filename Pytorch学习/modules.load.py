import torch
import torchvision

# 加载网络模型
module1 = torch.load("vgg16.pth")
print(module1)

# 加载网络模型参数
module2 = torchvision.models.vgg16(pretrained=False)
module2.load_state_dict(torch.load("vgg16_state_dict.pth"))
print(module2)
