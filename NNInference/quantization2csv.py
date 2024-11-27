import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np

# from ParamsReader import params_reader
from quantization_util import *


# 2层网络 FNCC
layer_1 = NNLayer()
layer_2 = NNLayer()

# layer_1
MNIST_dataset = MNIST(root='./FCNNModelCreater/MNIST/data', train=True, download=False, transform=transforms.ToTensor())
image, label = MNIST_dataset[0] 
layer_1.input = image.squeeze().numpy().flatten()  # layer_1.input








