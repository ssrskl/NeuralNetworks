from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('logs')
image_path = r'C:\Codes\PythonProjects\FirstTest\data\images\test.png'
img_PIL = Image.open(image_path)
# 转换成numpy数组
img_array = np.array(img_PIL)
writer.add_image("test", img_array, 1, dataformats='HWC')

writer.close()
