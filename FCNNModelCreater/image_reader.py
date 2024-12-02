import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
from PIL import Image
import os



def get_image_data():
    MNIST_dataset = MNIST(root='./FCNNModelCreater/MNIST/data', train=True, download=False, transform=transforms.ToTensor())
    image, label = MNIST_dataset[4] 

     # 将图像从 Tensor 转换回 PIL 图像
    image_pil = transforms.ToPILImage()(image)
    
    # 保存图像为 .png 文件
    save_path = './FCNNModelCreater'
    os.makedirs(save_path, exist_ok=True)  # 创建目录（如果不存在）
    
    image_filename = f"image_{label}.png"  # 使用标签作为文件名
    image_pil.save(os.path.join(save_path, image_filename))
    
    print(label)
    return image.squeeze().numpy().flatten()  # layer_1.input

    


# image = get_image_data()
# print(image)

# img = Image.fromarray(image_uint8)
# img.save("mnist_image.png")