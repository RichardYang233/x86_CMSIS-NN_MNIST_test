import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from Quantification.quantizer import *
from Quantification.data_utils import *
from Quantification.param_parser import *


def output_quantized_dataset_2_csv(output_path: str, drt_type: str):

    MNIST_dataset = MNIST(root='./FCNNModelCreater/MNIST/data', train=False, download=False, transform=transforms.ToTensor())
    dataset_length = len(MNIST_dataset)

    with open(output_path, mode='w', newline='') as csvfile:

        count = 0    
        for i in range(dataset_length):
            image_data= get_single_MNIST_test_dataset(MNIST_dataset, i) # 读取
            quantized_data = quantize(image_data, drt_type) # 量化
            quantized_data_with_label= np.insert(quantized_data, 0, MNIST_dataset[i][1])
            write_csv_row(csvfile, quantized_data_with_label)  # 输出
            count += 1

        print(f'Successfully quantize {count} test-dataset to csv !') 


def output_quantized_params_2_csv(output_path: str, label: str, drt_type: str):

    raw_params = read_params(output_path, label)
    quantized_data = quantize(raw_params, drt_type)
    replace_params(output_path, label, quantized_data)
    
    print(f"Successfully wirte {label} with type {drt_type} !")

    