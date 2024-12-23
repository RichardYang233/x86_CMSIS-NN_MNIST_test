import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import csv
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Quantification.quantizer import *


MNIST_TEST_DATASET_FILE = './FCNNModelCreater/MNIST_test_dataset.csv'


def output_quantized_dataset_2_csv(outputFile: str, drt_type: str):

    # 获取单个数据
    def get_single_MNIST_test_dataset(MNIST_dataset: MNIST, i: int) -> np.ndarray: 
        """
        Args:
            MNIST_dataset(MNIST): 需要转换为tensor格, 如下所示
            MNIST_dataset = MNIST(root='./FCNNModelCreater/MNIST/data', train=False, download=False, transform=transforms.ToTensor())
        """
        image, label = MNIST_dataset[i] # 获取图像数据
        return image.squeeze().numpy().flatten()
    
    # 输出单个数据
    def output_single_dataset_2_csv(csvfile, data: np.ndarray):
        writer = csv.writer(csvfile)
        writer.writerow(data)


    MNIST_dataset = MNIST(root='./FCNNModelCreater/MNIST/data', train=False, download=False, transform=transforms.ToTensor())
    dataset_length = len(MNIST_dataset)

    with open(outputFile, mode='w', newline='') as csvfile:

        count = 0    
        for i in range(dataset_length):
            image_data= get_single_MNIST_test_dataset(MNIST_dataset, i) # 读取
            quantized_data = quantize(image_data, drt_type) # 量化
            output_single_dataset_2_csv(csvfile, quantized_data) # 输出
            count += 1

        print(f'Successfully quantize {count} test-dataset to csv !') 

    

# -- 使用示例 -- #
# output_quantized_dataset_2_csv(MNIST_TEST_DATASET_FILE)
# print('run success !!!')

    