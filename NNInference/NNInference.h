#ifndef __NN_INFERENCE_H
#define __NN_INFERENCE_H

#include "arm_nnfunctions.h"
#include "main.h"


// 全连接网络参数
#define INPUT_SIZE 784  // 输入尺寸 28x28
#define HIDDEN_SIZE 512 // 隐藏层节点数
#define OUTPUT_SIZE 10  // 输出类别


// CMSIS-NN 参数结构
cmsis_nn_context ctx;
cmsis_nn_fc_params fc_params;
cmsis_nn_per_tensor_quant_params quant_params;
cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

// 输入层 (图像) (不符合输入变量类型，暂用)
extern int8_t input[INPUT_SIZE];

// 隐藏层
int8_t hidden_weights[HIDDEN_SIZE * INPUT_SIZE]; // 权重
int32_t hidden_bias[HIDDEN_SIZE];               // 偏置
int8_t hidden_output[HIDDEN_SIZE];              // 输出

// 输出层
int8_t output_weights[OUTPUT_SIZE * HIDDEN_SIZE];
int32_t output_bias[OUTPUT_SIZE];
int8_t output[OUTPUT_SIZE];


void init_nn_params(void);
void run_inference(void);


#endif