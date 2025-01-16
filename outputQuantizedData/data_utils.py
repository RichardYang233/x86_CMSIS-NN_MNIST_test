#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import numpy as np
import csv
from .csv_utils import *


MNIST_addr = './dataset_and_model/MNIST/data'


# ---------- MNIST ---------- #

def get_MNIST_data(MNIST_addr: str): 
    data = MNIST(root=MNIST_addr, train=False, download=False, transform=transforms.ToTensor())
    length = len(data)
    return data, length

def get_1_image_data(MNIST_data: MNIST, cnt: int) -> np.array:
    """
    Args:
        MNIST_dataset(MNIST): 需要转换为tensor格式, 如下所示
        MNIST_dataset = MNIST(root='./FCNNModelCreater/MNIST/data', train=False, download=False, transform=transforms.ToTensor())
    """
    image, label = MNIST_data[cnt]
    return image.squeeze().numpy().flatten()

# ---------- params ---------- #

def read_params(file_path, title) -> np.array:

    def find_title(reader, title) -> int:
        for row in reader:
            if row[0] == title:
                return row
        raise ValueError(f"{title} not found in the CSV file.")
    
    def analysis_shape(row):
        shape =  eval(row[1].replace("torch.Size", "")) # 转换字符串为元组
        return shape
    
    def read_params(reader, shape):
        params = []
        while True:
            try:
                params_row = next(reader) # 读取下一行
                params.extend([float(x) for x in params_row]) # 将每行的数据转换为浮点数并追加到列表
                # 停止条件：数据行长度等于期望的总数据量
                if len(params) == np.prod(shape):
                    return params
            except StopIteration:
                raise ValueError("Unexpected end of file while reading data.")
            
    with open(file_path, mode='r') as file:
        
        reader = csv.reader(file)

        row = find_title(reader, title)
        shape = analysis_shape(row)
        params = read_params(reader, shape)
        params_reshaped = np.array(params).reshape(shape) # 将数据重塑为目标形状

        return params_reshaped


def replace_params(file_path, title, q_data):
    
    with open(file_path, mode='r', newline='') as file:

        reader = csv.reader(file)
        data = list(reader)

    # 查找目标行索引
    target_index = None
    for i, row in enumerate(data):
        if title in row:
            target_index = i
            break
    
    # 替换数据
    if target_index is not None:
        for i in range(q_data.shape[0]):
            if q_data.ndim == 1:
                row = q_data
                data[target_index + 1] = row
            else:   
                row = q_data[i, :].tolist()
                data[target_index + 1 + i] = row # 插入新数据行
        with open(file_path, mode='w', newline='') as f:
            write_csv_rows(f, data)
    else:
        print("没有找到目标行 ！！！")




