from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer = SummaryWriter('logs')
img_path = r'C:\Codes\PythonProjects\FirstTest\data\images\test.png'
img = Image.open(img_path)
# 转换成tensor
trans_totensor = transforms.ToTensor()
tensor_img = trans_totensor(img)
print(tensor_img)

# Resize
print(img.size)
trans_resize = transforms.Resize((100, 100))
img_resize = trans_resize(img)
img_tensor = trans_totensor(img_resize)
writer.add_image("Resize", img_tensor, 1, dataformats='CHW')

# Compose
trans_resize_2 = transforms.Resize((200, 200))
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Compose", img_resize_2, 1, dataformats='CHW')
writer.close()