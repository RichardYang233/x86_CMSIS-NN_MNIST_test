from Quantification.paramParser import *
from Quantification.quantizer import *
from FCNNModelCreater.image_reader import *


# --- CSV --- #

SRC_PATH = './FCNNModelCreater/params.csv'
DRT_PATH = './NNInference/quantized_params.csv'

copy_csv(SRC_PATH, DRT_PATH)

# --- Quantizer --- #

quantizer = Quantizer()

# --- ParamParser --- #

parser = ParamParser()

# --- 流程 --- #

def process(label, drt_type):
    raw_data       = parser.read_params(DRT_PATH, label)
    quantized_data = quantizer.quantitate(raw_data, drt_type)
    parser.write_params(quantized_data)
    print(f"Successfully wirte {label} with type {drt_type} !")

# - 量化图像数据 - #

raw_data = get_image_data()
quantized_data = quantizer.quantitate(raw_data, 'int8')
str_data = ",".join(map(str, quantized_data))
print(str_data)

# - 逐层量化参数 - #

process('fc1.weight', 'int8')

process('fc1.bias', 'int8')

process('fc2.weight', 'int8')

process('fc2.bias', 'int8')


# --- lay_1.input --- #

# layer_1.input = get_image_data()
# print(layer_1.src_path)



print("run successful !!!")
