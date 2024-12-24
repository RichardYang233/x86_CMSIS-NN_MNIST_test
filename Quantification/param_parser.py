import csv
import numpy as np

from Quantification.quantizer import *
from Quantification.data_utils import *


# ===================================== #
#                 示例                  #
# ===================================== #

# label,"torch.Size([shape])""
# params...

# fc1.weight,"torch.Size([512, 784])"
# x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x
# x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x


def find_label(label, file):
    reader = csv.reader(file)
    for row in reader:
        if row[0] == label:
            return True
    raise ValueError(f"{label} not found in the CSV file.")
            

def read_params(file_path, label) -> np.array:
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        # 逐行检查
        for row in reader:
            if row[0] == label:  # 比对目标名称
                # 解析维度信息
                shape = eval(row[1].replace("torch.Size", "")) # 转换字符串为元组

                # 读取数据部分
                data = []
                while True:
                    try:
                        data_row = next(reader)  # 读取下一行
                        data.extend([float(x) for x in data_row]) # 将每行的数据转换为浮点数并追加到列表
                        # 停止条件：数据行长度等于期望的总数据量
                        if len(data) == np.prod(shape):
                            break
                    except StopIteration:
                        raise ValueError("Unexpected end of file while reading data.")
                    
                # 将数据重塑为目标形状
                data = np.array(data).reshape(shape)
                return data # return shape
    raise ValueError(f"{label} not found in the CSV file.")

def replace_params(file_path, label, quantized_data):
    with open(file_path, mode='r' , newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
    # 查找目标行索引
    target_index = None
    for i, row in enumerate(data):
        if label in row:
            target_index = i
            break
    # 替换数据
    if target_index is not None:
        for i in range(quantized_data.shape[0]):
            if quantized_data.ndim == 1:
                row = quantized_data
                data[target_index + 1] = row
            else:   
                row = quantized_data[i, :].tolist()
                data[target_index + 1 + i] = row # 插入新数据行
        with open(file_path, mode='w', newline='') as f:
            write_csv_rows(f, data)
            
    else:
        print("没有找到目标行 ！！！")


# class ParamParser():

#     def __init__(self, drt_path: str = None, label: str = None):
#         self.file_path = drt_path
#         self.label = label

#     def set_parser(self, drt_path: str, label: str):
#         self.file_path = drt_path
#         self.label = label

#     def set_label(self, label: str):
#         self.label = label

#     def read_params(self) -> np.array:
#         with open(self.file_path, mode='r') as file:
#             reader = csv.reader(file)
#             # 逐行检查
#             for row in reader:
#                 if row[0] == self.label:  # 比对目标名称
#                     # 解析维度信息
#                     shape = eval(row[1].replace("torch.Size", "")) # 转换字符串为元组
#                     # 读取数据部分
#                     data = []
#                     while True:
#                         try:
#                             data_row = next(reader)  # 读取下一行
#                             data.extend([float(x) for x in data_row]) # 将每行的数据转换为浮点数并追加到列表
#                             # 停止条件：数据行长度等于期望的总数据量
#                             if len(data) == np.prod(shape):
#                                 break
#                         except StopIteration:
#                             raise ValueError("Unexpected end of file while reading data.")
#                     # 将数据重塑为目标形状
#                     data = np.array(data).reshape(shape)
#                     return data # return shape
#         raise ValueError(f"{self.label} not found in the CSV file.")




# def read_params(file_path, label): 

#     parser = ParamParser()
#     parser.set_parser(file_path, label)
#     return parser.read_params()

# def replace_params(file_path, label, quantized_data):

#     parser = ParamParser()
#     parser.set_parser(file_path, label)
#     parser.replace_params(quantized_data)


