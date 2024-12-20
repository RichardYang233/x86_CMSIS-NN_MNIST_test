from Quantification.paramParser import *
from Quantification.quantizer import *
from FCNNModelCreater.image_reader import *


# --- CSV --- #

SRC_PATH = './FCNNModelCreater/params.csv'
DRT_PATH = './NNInference/quantized_params.csv'

copy_csv(SRC_PATH, DRT_PATH)

# --- 量化处理 --- #

def read_raw_params(label): 
    parser = ParamParser()
    parser.set_parser(DRT_PATH, label)
    raw_data = parser.read_params()
    return raw_data

def quantitate_raw_params(raw_data, drt_type):
    quantizer = Quantizer()
    quantizer.set_quantizer(raw_data, drt_type)
    quantized_data = quantizer.quantitate()
    return quantized_data

def write_params(quantized_data, label):
    parser = ParamParser()
    parser.set_parser(DRT_PATH, label)
    parser.write_params(quantized_data)

def process(label, drt_type):
    raw_data = read_raw_params(label)
    quantized_data = quantitate_raw_params(raw_data, drt_type) 
    write_params(quantized_data, label)
    print(f"Successfully wirte {label} with type {drt_type} !")

# - 量化图像数据 - #

raw_data = get_image_data()
quantizer = Quantizer()
quantizer.set_quantizer(raw_data, 'int8')
quantized_data = quantizer.quantitate()
str_data = ",".join(map(str, quantized_data))
print('图像像素数据为：', str_data)

# - 逐层量化参数 - #

process('fc1.weight', 'int8')

process('fc1.bias', 'int8')

process('fc2.weight', 'int8')

process('fc2.bias', 'int8')


print("run successful !!!")
