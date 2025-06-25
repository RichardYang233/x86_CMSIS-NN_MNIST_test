#include <stdio.h>

#include "arm_nnfunctions.h"

#include "cmsis_nn_helper.h"
#include "config.h"
#include "param.h"


// ----------------------------------------------------------------------
// 全连接层
// ----------------------------------------------------------------------

/**
 * @brief 设置全连接层的 上下文信息
 */
void fc_set_context(fc_model_handle *fc_model)
{
    fc_model->ctx.buf = NULL;
    fc_model->ctx.size = 0;
}

/**
 * @brief 设置全连接层的 维度信息
 * @param input_dim 该层输入维度
 * @param output_dim 该层输出维度
 */
void fc_set_dims(fc_model_handle *fc_model, int32_t input_dim, int32_t output_dim)
{
    // 输入维度
    fc_model->input_dims.n = 1;
    fc_model->input_dims.h = 1;
    fc_model->input_dims.w = 1;
    fc_model->input_dims.c = input_dim;
    // 权重维度
    fc_model->weight_dims.n = input_dim;
    fc_model->weight_dims.h = 1;
    fc_model->weight_dims.w = 1;
    fc_model->weight_dims.c = output_dim;
    // 偏置维度
    fc_model->bias_dims.n = 1;
    fc_model->bias_dims.h = 1;
    fc_model->bias_dims.w = 1;
    fc_model->bias_dims.c = output_dim;
    // 输出维度
    fc_model->output_dims.n = 1;
    fc_model->output_dims.h = 1;
    fc_model->output_dims.w = 1;
    fc_model->output_dims.c = output_dim; 
}

/**
 * @brief 设置全连接层的 偏移 和 数据范围 信息
 * @param offset 输出偏移
 * @param min 最小值
 * @param max 最大值
 */
void fc_set_fc_params(fc_model_handle *fc_model, int32_t offset, int32_t min, int32_t max)
{
    fc_model->fc_params.input_offset = 0;       
    fc_model->fc_params.filter_offset = 0;      
    fc_model->fc_params.output_offset = offset;      
    fc_model->fc_params.activation.min = min;     
    fc_model->fc_params.activation.max = max;   
}


/**
 * @brief 设置全连接层的 反量化参数
 */
void fc_set_quant_params(fc_model_handle *fc_model, int32_t multiplier, int32_t shift)
{
    fc_model->quant_params.multiplier = multiplier;
    fc_model->quant_params.shift = shift;
}


cmsis_nn_conv_params conv_set_params_s8(int32_t input_offset, int32_t output_offset, int32_t stride, int32_t padding)
{
    cmsis_nn_conv_params conv_params;

    conv_params.activation.max = INT8_MAX;
    conv_params.activation.min = INT8_MIN;
    conv_params.input_offset = input_offset;
    conv_params.input_offset = output_offset;
    conv_params.stride.w = stride;
    conv_params.stride.h = stride;
    conv_params.padding.w = padding;
    conv_params.padding.h = padding;
    // NOTE: dilation 默认值为 1，暂不清楚具体用法
    conv_params.dilation.w = 1; 
    conv_params.dilation.h = 1;

    return conv_params;
}






