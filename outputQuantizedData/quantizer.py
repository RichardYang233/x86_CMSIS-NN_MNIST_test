#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def get_scale(array, drt_type) -> float:

    max_val = np.amax(array) 
    min_val = np.amin(array)
    if (drt_type == 'int8'):
        return max(abs(max_val), abs(min_val)) / (2**7 - 1)
    elif(drt_type == 'int32'):
        return min(abs(max_val), abs(min_val)) / (2**31 - 1) # 原本 bais 使用，但会数据溢出
    else:
        raise ValueError(f"Unsupported data type: {drt_type}. Valid types are 'int8' and 'int32'.")

def get_zero_point(array, drt_type) -> int:

    scale = get_scale(array, drt_type)
    min_val = np.amin(array)
    return round(- min_val / scale)


def quantize(array, drt_type) -> np.int8 | np.int32:
    
    scale = get_scale(array, drt_type)
    zero_point = get_zero_point(array, drt_type)
    drt_data = np.around( (array - np.amin(array)) / scale + zero_point )

    return drt_data.astype(np.int8)     # NOTE: 这里写死了