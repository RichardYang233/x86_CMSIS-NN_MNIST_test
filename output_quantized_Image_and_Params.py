from Quantification.paramParser import *
from Quantification.quantizer import *
from FCNNModelCreater.test_dataset_reader import *


# - 复制 原始数据 到 新文件 - #

SRC_PATH = './FCNNModelCreater/params.csv'
DRT_PATH = './NNInference/quantized_params.csv'

copy_csv(SRC_PATH, DRT_PATH)

# - 量化训练集数据 - #

TEST_DATESET_FILE = './NNInference/quantized_test_dataset.csv'

output_quantized_dataset_2_csv(TEST_DATESET_FILE, 'int8')

# - 逐层量化参数 - #

output_quantized_params_2_csv(DRT_PATH, 'fc1.weight', 'int8')

output_quantized_params_2_csv(DRT_PATH, 'fc1.bias'  , 'int8')

output_quantized_params_2_csv(DRT_PATH, 'fc2.weight', 'int8')

output_quantized_params_2_csv(DRT_PATH, 'fc2.bias'  , 'int8')



print("run successful !!!")
