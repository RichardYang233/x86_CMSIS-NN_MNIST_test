#include <stdio.h>
#include <stdint.h>
#include "tensor_utils.h"

/**
 * @brief CMSIS-NN tensor 的维度顺序为 HWC
 *        PyTorch tensor 的维度顺序为 CHW
 *        此函数用于将 HWC 转为 CHW 来输出，以便观察数据
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