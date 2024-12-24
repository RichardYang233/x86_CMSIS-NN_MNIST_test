import csv
import shutil

from torchvision.datasets import MNIST
import numpy as np


# ===================================== #
#                 CSV                   #
# ===================================== #

# -- read -- #

# TODO

# -- write -- #

def write_csv_rows(csvfile, data):
    writer = csv.writer(csvfile)
    writer.writerows(data)

def write_csv_row(csvfile, data):
    writer = csv.writer(csvfile)
    writer.writerow(data)

# -- file -- # 

def copy_csv_file(src_path, drt_path): 
    shutil.copy(src_path, drt_path)
    print(f"File copied from {src_path} to {drt_path}")


# ===================================== #
#                MNIST                  #
# ===================================== #

# 获取单个 MNIST 测试集数据
def get_single_MNIST_test_dataset(MNIST_dataset: MNIST, i: int) -> np.ndarray: 
    """
    Args:
        MNIST_dataset(MNIST): 需要转换为tensor格式, 如下所示
        MNIST_dataset = MNIST(root='./FCNNModelCreater/MNIST/data', train=False, download=False, transform=transforms.ToTensor())
    """
    image, label = MNIST_dataset[i] # 获取图像数据
    return image.squeeze().numpy().flatten()