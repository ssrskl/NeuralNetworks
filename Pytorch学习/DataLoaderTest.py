import torchvision
from torch.utils.data import DataLoader

# 准备数据集
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# 准备数据集加载器
test_data_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

img, target = test_data[0]
print(img.shape)
print(target)

for data in test_data_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)

