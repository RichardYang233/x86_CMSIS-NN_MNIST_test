#include <stdio.h>
#include <stdint.h>
#include "tensor_utils.h"

/**
 * @brief CMSIS-NN tensor 的维度顺序为 HWC
 *        PyTorch tensor 的维度顺序为 CHW
 *        此函数用于将 HWC 转为 CHW 来输出，以便观察数据
 *        卷积层的输入输出需要通过此函数print
 * @param tensor CMSIS-NN HWC tensor
 * @param height H
 * @param width W
 * @param channel C
 */
void print_CHW_from_HWC(int8_t* tensor, int32_t height, int32_t width, int32_t channel)
{
    int8_t* ptr = &tensor[0];

    printf("----- [H]: %d [W]: %d [C]: %d ----- \n", height, width, channel);
    for(int i = 0; i < height * width; i++)
    {
        printf("%4d ", *ptr);
        ptr = ptr + channel;
        if(( i + 1) % width == 0) printf("\n");
    }
}

/**
 * @brief CMSIS-NN tensor 的维度顺序为 HWC
 *        PyTorch tensor 的维度顺序为 CHW
 *        此函数用于将 HWC 转为 CHW 
 *        将 CMSIS-NN 的卷积层输出到全连接层之前，需进行此操作
 * @param tensor CMSIS-NN HWC tensor
 * @param height H
 * @param width W
 * @param channel C
 */
void convert_HWC_2_CHW(int8_t* tensor, int32_t height, int32_t width, int32_t channel)
{
    int32_t total_size = height * width * channel;

    int8_t temp[total_size];
    int32_t idx = 0;

    for (int c = 0; c < channel; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                temp[idx++] = tensor[(h * width + w) * channel + c];
            }
        }
    }
    
    for(int i = 0; i < 4*4*16; i++)
    {
        tensor[i] = temp[i];
    }
}

