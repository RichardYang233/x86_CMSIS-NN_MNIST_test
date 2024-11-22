import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from PIL import Image

transform = transforms.ToTensor()
mnist_dataset = MNIST(root='./FCNNModelCreater/MNIST/data', train=True, download=False, transform=transforms.ToTensor())

# 提取第一个样本 (5)
image, label = mnist_dataset[0]


print("Image shape:", image.shape)  # 图像为 (1, 28, 28)
print("Label:", label)
image_2d = image.squeeze().numpy()
image_uint8 = (image_2d * 255).astype('uint8')
flattened_data = image_uint8.flatten()
flattened_str = ",".join(map(str, flattened_data))
print("Image Data:", flattened_str)



# img = Image.fromarray(image_uint8)
# img.save("mnist_image.png")