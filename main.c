#include <stdio.h>
// #include "arm_math.h"
// #include "arm_nn_types.h"
#include "main.h"
#include "arm_nnfunctions.h"
#include "params_reader.h"


// 输入层 (图像)
uint8_t input[INPUT_SIZE] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126, 136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253, 225, 172, 253, 242, 195, 64, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253, 253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0, 43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139,
                             253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171, 219, 253,
                             253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244,
                             133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// 隐藏层 
int8_t hidden_weights[HIDDEN_SIZE][INPUT_SIZE]; // 权重
int32_t hidden_bias[HIDDEN_SIZE];               // 偏置
int8_t hidden_output[HIDDEN_SIZE];              // 输出

// 输出层
int8_t output_weights[OUTPUT_SIZE][HIDDEN_SIZE];
int32_t output_bias[OUTPUT_SIZE];
int8_t output[OUTPUT_SIZE];

// CMSIS-NN 参数结构体
cmsis_nn_context ctx;
cmsis_nn_fc_params fc_params;
cmsis_nn_per_tensor_quant_params quant_params;
cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;





// 初始化 CMSIS-NN 参数
void init_nn_params() {






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
void run_inference() {
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




int main(void) {
    // 初始化参数
    init_nn_params();

    // 执行推理
    run_inference();

    // 输出分类结果
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("Class %d score: %d\n", i, output[i]);
    }

    return 0;
}



