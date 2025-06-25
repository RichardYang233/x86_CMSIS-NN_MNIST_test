#ifndef __PARAM_H
#define __PARAM_H

#include "arm_nnfunctions.h"
#include "config.h"

/* ----------------------------------------------------------- */
/* 模型参数                                                     */
/* ----------------------------------------------------------- */

/**
 * @brief 全连接 参数句柄
 */
typedef struct {
    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims, weight_dims, bias_dims, output_dims;
} fc_model_handle;

/**
 * @brief 卷积 参数句柄
 */
typedef struct {
    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;
} cnn_model_handle;

/**
 * @brief 池化 参数句柄
 */
typedef struct {
    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    int8_t src;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;
    int8_t dst;
} pool_model_handle;


#endif