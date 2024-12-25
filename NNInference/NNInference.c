#include "main.h"
#include <stdio.h>
#include "NNInference.h"
#include "arm_nnfunctions.h"


// 输入图像数据
// int8_t input[INPUT_SIZE] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,27,74,105,126,126,56,43,74,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,43,116,126,126,94,105,126,126,126,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,28,121,126,95,32,2,6,91,126,126,58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,126,126,91,7,0,0,46,126,126,112,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,66,126,126,73,7,0,0,0,107,126,126,39,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,126,123,88,4,0,0,4,39,122,126,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,116,126,88,0,0,0,18,100,126,126,84,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,126,126,15,11,59,98,120,126,126,125,38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,115,126,126,126,126,126,113,113,126,115,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,27,117,126,108,69,21,12,96,126,71,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,127,126,54,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,126,126,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,126,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,126,126,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,53,126,126,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,127,126,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,109,126,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,126,94,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,92,126,85,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,73,126,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


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