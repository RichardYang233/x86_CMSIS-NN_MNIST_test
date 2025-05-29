#include <stdio.h>

#include "arm_nnfunctions.h"

#include "cmsis_nn_helper.h"
#include "config.h"
#include "param.h"


/**
 * @brief 设置全连接层的 上下文信息
 */
void FC_set_context(cmsis_nn_context *ctx)
{
    ctx->buf = NULL;
    ctx->size = 0;
}

/**
 * @brief 设置全连接层的 维度信息
 * @param input_dim 该层输入维度
 * @param output_dim 该层输出维度
 */
void FC_set_dims(int32_t input_dim, int32_t output_dim)
{
    // 输入维度
    input_dims.n = 1;
    input_dims.h = 1;
    input_dims.w = 1;
    input_dims.c = input_dim;
    // 权重维度
    weight_dims.n = input_dim;
    weight_dims.h = 1;
    weight_dims.w = 1;
    weight_dims.c = output_dim;
    // 偏置维度
    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dim;
    // 输出维度
    output_dims.n = 1;
    output_dims.h = 1;
    output_dims.w = 1;
    output_dims.c = output_dim; 
}

/**
 * @brief 设置全连接层的 偏移 和 数据范围 信息
 * @param offset 输出偏移
 * @param min 最小值
 * @param max 最大值
 */
void FC_set_fc_params(cmsis_nn_fc_params *fc_params, int32_t offset, int32_t min, int32_t max)
{
    fc_params->input_offset = 0;       
    fc_params->filter_offset = 0;      
    fc_params->output_offset = offset;      
    fc_params->activation.min = min;     
    fc_params->activation.max = max;   
}


/**
 * @brief 设置全连接层的 反量化参数
 */
void FC_set_quant_params(cmsis_nn_per_tensor_quant_params *quant_params, int32_t multiplier, int32_t shift)
{
    quant_params->multiplier = multiplier;
    quant_params->shift = shift;
}




