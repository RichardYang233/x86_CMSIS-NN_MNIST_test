import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np



def get_image_data():
    # 生成图像量化数据
    MNIST_dataset = MNIST(root='./FCNNModelCreater/MNIST/data', train=True, download=False, transform=transforms.ToTensor())
    image, label = MNIST_dataset[2] 
    return image.squeeze().numpy().flatten()  # layer_1.input

    # layer_1.quantitate_params()
    # show(layer_1.input)