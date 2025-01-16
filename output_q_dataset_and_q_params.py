#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from outputQuantizedData import *


def output_q_dataset(output_path: str, drt_type: str):

    MNIST_data, MINIST_length = get_MNIST_data('./dataset_and_model/MNIST/data') 

    with open(output_path, mode='w', newline='') as csvfile:
        count = 0    
        for i in range(MINIST_length):
            image_data = get_1_image_data(MNIST_data, i) # 读取一个图像
            q_image_data = quantize(image_data, drt_type) # 量化
            q_image_data_with_label= np.insert(q_image_data, 0, MNIST_data[i][1])
            write_csv_row(csvfile, q_image_data_with_label)  # 输出
            count += 1
        print(f'Successfully quantize {count} test-dataset to csv !') 


def output_q_params(output_path: str, label: str, drt_type: str):

    raw_params = read_params(output_path, label)
    q_params = quantize(raw_params, drt_type)
    replace_params(output_path, label, q_params)
    
    print(f"Successfully wirte {label} with type {drt_type} !")



if __name__ == "__main__":

    # - 量化训练集数据 - #

    TEST_DATESET_FILE = './data/q_dataset.csv'

    output_q_dataset(output_path=TEST_DATESET_FILE, drt_type='int8')

    # - 逐层量化参数 - #

    SRC_PATH = './data/.net_params.csv'
    DRT_PATH = './data/q_params.csv'

    copy_csv_file(SRC_PATH, DRT_PATH)

    output_q_params(DRT_PATH, 'hidden_layer.weight', 'int8')

    output_q_params(DRT_PATH, 'hidden_layer.bias'  , 'int8')

    output_q_params(DRT_PATH, 'output_layer.weight', 'int8')

    output_q_params(DRT_PATH, 'output_layer.bias'  , 'int8')

    print("run successful !!!")
