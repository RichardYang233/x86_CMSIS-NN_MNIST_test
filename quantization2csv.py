import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np

from ParamsReader.params_transfer import *
from NNInference.quantization_util import *


# 2层网络 FNCC
layer_1 = NNLayer()
layer_2 = NNLayer()

# # 生成图像量化数据
MNIST_dataset = MNIST(root='./FCNNModelCreater/MNIST/data', train=True, download=False, transform=transforms.ToTensor())
image, label = MNIST_dataset[2] 
layer_1.input = image.squeeze().numpy().flatten()  # layer_1.input

layer_1.quantitate_params()
show(layer_1.input)

SRC_PATH = './FCNNModelCreater/params.csv'
DRT_PATH = './NNInference/quantized_params.csv'

CSVHandle = CSVHandler(SRC_PATH, DRT_PATH)

# fc1.weight
CSVHandle.set_label('fc1.weight')
layer_1.weight = CSVHandle.read_params()
layer_1.quantitate_params()
CSVHandle.output_quantized_params(layer_1.weight)

# fc1.bias
CSVHandle.set_label('fc1.bias')
layer_1.bias = CSVHandle.read_params()
layer_1.quantitate_params()
CSVHandle.output_quantized_params(layer_1.bias)

# fc2.weight
CSVHandle.set_label('fc2.weight')
layer_2.weight = CSVHandle.read_params()
layer_2.quantitate_params()
CSVHandle.output_quantized_params(layer_2.weight)

# fc2.bias
CSVHandle.set_label('fc2.bias')
layer_2.bias = CSVHandle.read_params()
layer_2.quantitate_params()
CSVHandle.output_quantized_params(layer_2.bias)

# copy_csv(SRC_PATH, DRT_PATH)
# shape, data = read_params(DRT_PATH, 'fc1.weight')
# layer_1.weight = data
# layer_1.quantitate_params()
# quant_params(DRT_PATH, layer_1.weight, 'fc1.weight')

# print(layer_1.weight)





