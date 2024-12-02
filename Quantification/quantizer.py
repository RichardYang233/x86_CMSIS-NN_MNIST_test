import numpy as np


class Quantizer:
    def __init__(self, array: np.array = None, drt_type: str = None):
        self.array = array
        self.drt_type = drt_type

    def set_quantizer(self, array, drt_type):
        self.array = array
        self.drt_type = drt_type

    def get_scale(self) -> float:
        max_val = np.amax(self.array) 
        min_val = np.amin(self.array)
        if (self.drt_type == 'int8'):
            return max(abs(max_val), abs(min_val)) / (2**7 - 1)
        elif(self.drt_type == 'int32'):
            return min(abs(max_val), abs(min_val)) / (2**31 - 1) # 原本 bais 使用，但会数据溢出
        else:
            raise ValueError(f"Unsupported data type: {self.drt_type}. Valid types are 'int8' and 'int32'.")

    def get_zero_point(self) -> int:
        scale = self.get_scale()
        min_val = np.amin(self.array)
        return round(- min_val / scale)

    def quantitate(self, array, drt_type) -> np.int8 | np.int32:
        self.array = array
        self.drt_type = drt_type
        scale = self.get_scale()
        zero_point = self.get_zero_point()
        drt_data = np.around( (self.array - np.amin(self.array)) / scale + zero_point )
        return drt_data.astype(np.int8)