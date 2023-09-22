import torchvision
import torch
from torch.nn import Linear

vgg16_true = torchvision.models.vgg16(pretrained=True)
vgg16_false = torchvision.models.vgg16(pretrained=False)

vgg16_true.add_module('add_linear', Linear(1000, 10))
vgg16_true.classifier[6] = Linear(4096, 4096)
print(vgg16_true)
