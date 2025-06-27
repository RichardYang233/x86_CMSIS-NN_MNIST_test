#include <stdio.h>
#include <stdlib.h>

#include "arm_nnfunctions.h"

#include "cmsis_nn_helper.h"
#include "config.h"
#include "param.h"


// ----------------------------------------------------------------------
// 全连接层
// ----------------------------------------------------------------------

void FC_SetParams(cmsis_nn_fc_params* fc_params , int32_t input_offset, int32_t filter_offset, int32_t output_offset)
{
    fc_params->activation.max = INT8_MAX;
    fc_params->activation.min = INT8_MIN;
    fc_params->input_offset = input_offset;
    fc_params->filter_offset = output_offset;
    fc_params->output_offset = output_offset;
}

void FC_SetQuant(cmsis_nn_per_tensor_quant_params* quant_params, int32_t multiplier, int32_t shift)
{
    quant_params->multiplier = multiplier;
    quant_params->shift = shift;
}

void FC_SetDims(cmsis_nn_dims* input_dims, int32_t input_size, cmsis_nn_dims* filter_dims, cmsis_nn_dims* output_dims, int32_t output_size)
{
    input_dims->n = 1;
    input_dims->w = 1;
    input_dims->h = 1;
    input_dims->c = input_size;

    filter_dims->n = input_size;
    filter_dims->c = output_size;

    output_dims->n = 1;
    output_dims->c = output_size;
}

void FC_Run(// const cmsis_nn_context *ctx,
            const cmsis_nn_fc_params *fc_params,
            const cmsis_nn_per_tensor_quant_params *quant_params,
            const cmsis_nn_dims *input_dims,
            const int8_t *input,
            const cmsis_nn_dims *filter_dims,
            const int8_t *kernel,
            const cmsis_nn_dims *bias_dims,
            const int32_t *bias,
            const cmsis_nn_dims *output_dims,
            int8_t *output)
{
    cmsis_nn_context ctx;
    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    arm_cmsis_nn_status status = arm_fully_connected_s8(&ctx,
                                                        fc_params, 
                                                        quant_params, 
                                                        input_dims,
                                                        input,
                                                        filter_dims,
                                                        kernel,
                                                        bias_dims,
                                                        bias,
                                                        output_dims,
                                                        output);
    if (status != ARM_CMSIS_NN_SUCCESS){ 
        printf("Convolution failed, status = %d\n", status); 
    }
    if (ctx.buf){
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
}

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


// ----------------------------------------------------------------------
// 卷积层 配置函数
// ----------------------------------------------------------------------

void CONV_Run(// const cmsis_nn_context *ctx,
              const cmsis_nn_conv_params *conv_params,
              const cmsis_nn_per_channel_quant_params *quant_params,
              const cmsis_nn_dims *input_dims,
              const int8_t *input_data,
              const cmsis_nn_dims *filter_dims,
              const int8_t *filter_data,
              const cmsis_nn_dims *bias_dims,
              const int32_t *bias_data,
              const cmsis_nn_dims *upscale_dims,
              const cmsis_nn_dims *output_dims,
              int8_t *output_data)
{
    cmsis_nn_context ctx;
    int32_t buf_size = arm_convolve_s8_get_buffer_size(input_dims, filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    arm_cmsis_nn_status status = arm_convolve_s8(&ctx,
                                                 conv_params,
                                                 quant_params,
                                                 input_dims, 
                                                 input_data,
                                                 filter_dims, 
                                                 filter_data,
                                                 bias_dims, 
                                                 bias_data,
                                                 NULL,
                                                 output_dims, 
                                                 output_data);
    if (status != ARM_CMSIS_NN_SUCCESS){ 
        printf("Convolution failed, status = %d\n", status); 
    }
    if (ctx.buf){
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
}

/**
 * @brief 
 * 
 * @param input_offset 
 * @param output_offset 
 * @param stride 
 * @param padding 
 * @return cmsis_nn_conv_params 
 */
void CONV_SetParams(cmsis_nn_conv_params* conv_params , int32_t input_offset, int32_t output_offset, int32_t stride, int32_t padding)
{
    conv_params->activation.max = INT8_MAX;
    conv_params->activation.min = INT8_MIN;
    conv_params->input_offset = input_offset;
    conv_params->output_offset = output_offset;
    conv_params->stride.w = stride;
    conv_params->stride.h = stride;
    conv_params->padding.w = padding;
    conv_params->padding.h = padding;
    // NOTE: dilation 默认值为 1，暂不清楚具体用法
    conv_params->dilation.w = 1; 
    conv_params->dilation.h = 1;
}

/**
 * @brief 
 */
/// TODO: 目前仅实现 per_channel 量化参数相同的情况
void CONV_SetQuant(cmsis_nn_per_channel_quant_params* per_channel_quant_params, 
                                int32_t channel,
                                int32_t* multiplier_arrry, 
                                int32_t multiplier, 
                                int32_t* shift_array,           
                                int32_t shift)
{
    for (int i = 0; i < channel; i++)
    {
        multiplier_arrry[i] = multiplier;
        shift_array[i] = shift;
    }

    per_channel_quant_params->multiplier = multiplier_arrry;
    per_channel_quant_params->shift = shift_array;
}

/**
 * @brief 
 * 
 */
/// NOTE: 无需设置 bias_dims，CMSIS-NN 卷积计算会用等价的 output_channel
void CONV_SetDims(cmsis_nn_dims* input_dims, int32_t input_size,  int32_t input_channel, 
                   cmsis_nn_dims* filter_dims, int32_t kernel_size, 
                   cmsis_nn_dims* output_dims, int32_t output_size,  int32_t output_channel)
{
    input_dims->n = 1;
    input_dims->w = input_size;
    input_dims->h = input_size;
    input_dims->c = input_channel;

    filter_dims->w = kernel_size;
    filter_dims->h = kernel_size;
    filter_dims->c = input_channel;

    output_dims->w = output_size;
    output_dims->h = output_size;
    output_dims->c = output_channel;
}

// ----------------------------------------------------------------------
// 池化层
// ----------------------------------------------------------------------

void MAXPOOL_Run(// const cmsis_nn_context *ctx,
                 const cmsis_nn_pool_params *pool_params,
                 const cmsis_nn_dims *input_dims,
                 const int8_t *src,
                 const cmsis_nn_dims *filter_dims,
                 const cmsis_nn_dims *output_dims,
                 int8_t *dst)
{
    cmsis_nn_context ctx;
    int32_t buf_size = arm_convolve_s8_get_buffer_size(input_dims, filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    arm_cmsis_nn_status status = arm_max_pool_s8(&ctx,
                                                 pool_params,
                                                 input_dims,
                                                 src,
                                                 filter_dims,
                                                 output_dims,
                                                 dst);
    if (status != ARM_CMSIS_NN_SUCCESS){ 
        printf("Convolution failed, status = %d\n", status); 
    }
    if (ctx.buf){
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
}

void POOL_SetParams(cmsis_nn_pool_params* pool_params, int32_t stride, int32_t padding)
{

    pool_params->activation.max = INT8_MAX;
    pool_params->activation.min = INT8_MIN;
    pool_params->stride.w = stride;
    pool_params->stride.h = stride;
    pool_params->padding.w = padding;
    pool_params->padding.h = padding;
}

/// NOTE: 无需设置 bias_dims，CMSIS-NN 卷积计算会用等价的 output_channel
void POOL_SetDims(cmsis_nn_dims* input_dims, int32_t input_size,  int32_t input_channel, 
                   cmsis_nn_dims* filter_dims, int32_t kernel_size, 
                   cmsis_nn_dims* output_dims, int32_t output_size,  int32_t output_channel)
{
    input_dims->n = 1;
    input_dims->w = input_size;
    input_dims->h = input_size;
    input_dims->c = input_channel;

    filter_dims->w = kernel_size;
    filter_dims->h = kernel_size;

    output_dims->w = output_size;
    output_dims->h = output_size;
    output_dims->c = output_channel;
}


// ----------------------------------------------------------------------
// 杂项
// ----------------------------------------------------------------------

int32_t get_argmax_result(int8_t* output, int32_t output_size)
{
    int8_t temp = 0;
    int8_t result = 0;

    for (int i = 0; i < output_size; i++)
    {   
        if (temp < output[i])
        {
            temp = output[i];
            result = i;
        }
    }
    return result;
}
