import numpy as np


class NNLayer(object):
    def __init__(self, input=None, weight=None, bias=None):
        self.input = input
        self.weight = weight
        self.bias = bias  

    def get_scale(self, dataArray, drtDataType: str) -> float:
        max_val = np.amax(dataArray) 
        min_val = np.amin(dataArray)
        if (drtDataType == 'int8'):
            return max(abs(max_val), abs(min_val)) / (2**7 - 1)
        elif(drtDataType == 'int32'):
            return min(abs(max_val), abs(min_val)) / (2**31 - 1) # 原本 bais 使用，但会数据溢出
        else:
            raise ValueError(f"Unsupported data type: {drtDataType}. Valid types are 'int8' and 'int32'.")

    def get_zero_point(self, dataArray, scale) -> int:
        min_val = np.amin(dataArray)
        return round(- min_val / scale)

    def quantitate(self, dataArray, drtDataType: str) -> np.int8 | np.int32:
        scale = self.get_scale(dataArray, drtDataType)
        zero_point = self.get_zero_point(dataArray, scale)
        drt_data = np.around( (dataArray - np.amin(dataArray)) / scale + zero_point )
        return drt_data.astype(np.int8)
    
    def quantitate_params(self):
        if self.input is not None:
            self.input = self.quantitate(self.input, 'int8')
        if self.weight is not None:
            self.weight = self.quantitate(self.weight, 'int8')
        if self.bias is not None:
            self.bias = self.quantitate(self.bias, 'int8') # 原本是 int32 ，但有溢出

def show(dataArray):
    str_data = ",".join(map(str, dataArray))
    print("Image Data:", str_data)


