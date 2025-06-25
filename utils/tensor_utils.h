#ifndef __TENSOR_UTILS_H
#define __TENSOR_UTILS_H

#include <stdint.h>

void print_CHW_from_HWC(int8_t* tensor, int32_t height, int32_t width, int32_t channel);
void convert_HWC_2_CHW(int8_t* tensor, int32_t height, int32_t width, int32_t channel);

#endif