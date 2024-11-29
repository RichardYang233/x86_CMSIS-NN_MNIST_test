#include "main.h"
#include <stdio.h>
#include "NNInference.h"
#include "arm_nnfunctions.h"


// 输入图像数据
int8_t input[INPUT_SIZE] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,116,19,0,0,0,0,0,0,0,0,0,31,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,90,19,0,0,0,0,0,0,0,0,0,63,81,0,0,0,0,0,0,0,0,0,0,0,0,0,1,76,105,20,0,0,0,0,0,0,0,0,0,110,81,0,0,0,0,0,0,0,0,0,0,0,0,0,13,127,81,0,0,0,0,0,0,0,0,0,0,111,81,0,0,0,0,0,0,0,0,0,0,0,0,0,91,127,62,0,0,0,0,0,0,0,0,0,23,122,81,0,0,0,0,0,0,0,0,0,0,0,0,0,99,127,28,0,0,0,0,0,0,0,0,0,60,127,81,0,0,0,0,0,0,0,0,0,0,0,0,11,115,127,14,0,0,0,0,0,0,0,0,0,79,127,60,0,0,0,0,0,0,0,0,0,0,0,0,81,127,108,8,0,0,0,0,0,0,0,0,0,79,127,33,0,0,0,0,0,0,0,0,0,7,43,89,124,127,45,0,0,0,0,0,0,0,0,0,0,79,127,42,0,0,0,23,24,58,72,75,120,121,117,89,120,126,20,0,0,0,0,0,0,0,0,0,0,75,126,118,103,103,103,126,127,125,120,99,71,45,14,2,116,125,0,0,0,0,0,0,0,0,0,0,0,0,59,88,88,88,88,88,49,28,0,0,0,0,0,51,127,110,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,127,68,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,127,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,127,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,127,47,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,127,48,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,127,76,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,127,76,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,127,76,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


// 初始化 CMSIS-NN 参数
void init_nn_params() 
{
    // 上下文设置（如果需要临时内存）
    ctx.buf = NULL;
    ctx.size = 0;

    // 全连接参数
    fc_params.input_offset = 0;       // 输入偏移
    fc_params.filter_offset = 0;      // 权重偏移
    fc_params.output_offset = 0;      // 输出偏移
    fc_params.activation.min = -128;  // 激活函数最小值（int8 范围）
    fc_params.activation.max = 127;   // 激活函数最大值（int8 范围）

    // 量化参数
    quant_params.multiplier = 1073741 ;// 1073741824; // 示例值，需根据量化导出结果调整
    quant_params.shift = 0;

    // 输入维度
    input_dims.n = 1;
    input_dims.h = 1;
    input_dims.w = 1;
    input_dims.c = INPUT_SIZE;

    // 隐藏层权重维度
    filter_dims.n = INPUT_SIZE;
    filter_dims.h = 1;
    filter_dims.w = 1;
    filter_dims.c = HIDDEN_SIZE;

    // 偏置维度
    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = HIDDEN_SIZE;

    // 隐藏层输出维度
    output_dims.n = 1;
    output_dims.h = 1;
    output_dims.w = 1;
    output_dims.c = HIDDEN_SIZE;
}

// 执行推理
void run_inference() 
{
    // 隐藏层
    arm_cmsis_nn_status status = arm_fully_connected_s8(
        &ctx,                 // 上下文
        &fc_params,           // 全连接参数
        &quant_params,        // 量化参数
        &input_dims,          // 输入维度
        input,                // 输入数据
        &filter_dims,         // 权重维度
        (int8_t *)hidden_weights, // 权重数据
        &bias_dims,           // 偏置维度
        hidden_bias,          // 偏置数据
        &output_dims,         // 输出维度
        hidden_output         // 隐藏层输出
    );

    if (status != ARM_CMSIS_NN_SUCCESS) {
        printf("Error: Fully connected layer computation failed.\n");
    }

    // 激活函数（ReLU）
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_output[i] = hidden_output[i] < 0 ? 0 : hidden_output[i];
    }

    // 输出层
    filter_dims.n = HIDDEN_SIZE; // 更新权重维度
    filter_dims.c = OUTPUT_SIZE;
    output_dims.c = OUTPUT_SIZE;

    status = arm_fully_connected_s8(
        &ctx,
        &fc_params,
        &quant_params,
        &bias_dims,         // 输入维度（来自隐藏层）
        hidden_output,        // 隐藏层输出
        &filter_dims,
        (int8_t *)output_weights,
        &output_dims,
        output_bias,
        &output_dims,
        output
    );

    if (status != ARM_CMSIS_NN_SUCCESS) {
        printf("Error: Output layer computation failed.\n");
    }
}