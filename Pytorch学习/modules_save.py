# 创建一个vgg16模型
import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=True)
# 保存模型-->保存模型的结构和参数
torch.save(vgg16, 'vgg16.pth')

# 保存模型-->只保存模型的参数
torch.save(vgg16.state_dict(), 'vgg16_state_dict.pth')
