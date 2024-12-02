import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np



def get_image_data():
    MNIST_dataset = MNIST(root='./FCNNModelCreater/MNIST/data', train=True, download=False, transform=transforms.ToTensor())
    image, label = MNIST_dataset[2] 

    return image.squeeze().numpy().flatten()  # layer_1.input


# image = get_image_data()
# print(image)

# img = Image.fromarray(image_uint8)
# img.save("mnist_image.png")