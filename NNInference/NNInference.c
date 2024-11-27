#include "main.h"
#include <stdio.h>
#include "NNInference.h"
#include "arm_nnfunctions.h"


// 输入图像数据
int8_t input[INPUT_SIZE] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,9,9,9,63,68,87,13,83,127,123,63,0,0,0,0,0,0,0,0,0,0,0,0,15,18,47,77,85,126,126,126,126,126,112,86,126,121,97,32,0,0,0,0,0,0,0,0,0,0,0,24,119,126,126,126,126,126,126,126,126,125,46,41,41,28,19,0,0,0,0,0,0,0,0,0,0,0,0,9,109,126,126,126,126,126,99,91,123,120,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,40,78,53,126,126,102,5,0,21,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,0,77,126,45,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,69,126,95,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,95,126,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,120,112,80,54,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,40,120,126,126,59,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,93,126,126,75,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,46,126,126,93,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,124,126,124,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,65,91,126,126,103,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,74,114,126,126,126,125,91,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,57,110,126,126,126,126,100,39,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,33,106,126,126,126,126,99,40,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,85,109,126,126,126,126,97,40,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,27,86,113,126,126,126,126,122,66,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,68,126,126,126,106,67,66,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


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
    quant_params.multiplier = 1073741824; // 示例值，需根据量化导出结果调整
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
        &filter_dims,         // 输入维度（来自隐藏层）
        hidden_output,        // 隐藏层输出
        &filter_dims,
        (int8_t *)output_weights,
        &bias_dims,
        output_bias,
        &output_dims,
        output
    );

    if (status != ARM_CMSIS_NN_SUCCESS) {
        printf("Error: Output layer computation failed.\n");
    }
}