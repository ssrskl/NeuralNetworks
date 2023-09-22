import torchvision
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')
# 声明一个转换器
dataset_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# 应用转换器
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transforms, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transforms, download=True)

print(train_set[0][0])
# 使用TensorBoard显示
writer.add_image("CIFAR10", train_set[0][0], 1, dataformats='CHW')
writer.close()