#ifndef __PARAM_H
#define __PARAM_H

#include "arm_nnfunctions.h"
#include "config.h"


// ----------------------------------------------------------------------------
// 全连接层
// ----------------------------------------------------------------------------

/* 设置 */
cmsis_nn_context ctx;
cmsis_nn_fc_params fc_params;
cmsis_nn_per_tensor_quant_params quant_params;
cmsis_nn_dims input_dims, weight_dims, bias_dims, output_dims;

/* 参数 */
// int8_t  input   []; // [input_size];
// int8_t  weight  []; // [input_size * output_size];
// int32_t bias    []; // [output_size];              
// int8_t  output  []; // [output_size];             


#endif