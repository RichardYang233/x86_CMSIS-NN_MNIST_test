#ifndef __CMSIS_NN_HELPER_H
#define __CMSIS_NN_HELPER_H

#include "param.h"





// ----------------------------------------------------------------------
// 全连接层
// ----------------------------------------------------------------------

void FC_SetParams(cmsis_nn_fc_params* fc_params , int32_t input_offset, int32_t filter_offset, int32_t output_offset);
void FC_SetQuant(cmsis_nn_per_tensor_quant_params* quant_params, int32_t multiplier, int32_t shift);
void FC_SetDims(cmsis_nn_dims* input_dims, int32_t input_size, cmsis_nn_dims* filter_dims, cmsis_nn_dims* output_dims, int32_t output_size);
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
            int8_t *output);

void fc_set_context(fc_model_handle *fc_model);
void fc_set_dims(fc_model_handle *fc_model, int32_t input_dim, int32_t output_dim);
void fc_set_fc_params(fc_model_handle *fc_model, int32_t offset, int32_t min, int32_t max);
void fc_set_quant_params(fc_model_handle *fc_model, int32_t multiplier, int32_t shift);

// ----------------------------------------------------------------------
// 卷积层
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
              int8_t *output_data);

void CONV_SetParams(cmsis_nn_conv_params* conv_params , int32_t input_offset, int32_t output_offset, int32_t stride, int32_t padding);
void CONV_SetQuant(cmsis_nn_per_channel_quant_params* per_channel_quant_params, int32_t channel, 
                              int32_t* multiplier_arrry, int32_t multiplier, 
                              int32_t* shift_array, int32_t shift);
void CONV_SetDims(cmsis_nn_dims* input_dims, int32_t input_size,  int32_t input_channel, 
                   cmsis_nn_dims* filter_dims, int32_t kernel_size, 
                   cmsis_nn_dims* output_dims, int32_t output_size,  int32_t output_channel);

// ----------------------------------------------------------------------
// 池化层
// ----------------------------------------------------------------------

void MAXPOOL_Run(// const cmsis_nn_context *ctx,
                 const cmsis_nn_pool_params *pool_params,
                 const cmsis_nn_dims *input_dims,
                 const int8_t *src,
                 const cmsis_nn_dims *filter_dims,
                 const cmsis_nn_dims *output_dims,
                 int8_t *dst);
void POOL_SetParams(cmsis_nn_pool_params* pool_params, int32_t stride, int32_t padding);
void POOL_SetDims(cmsis_nn_dims* input_dims, int32_t input_size,  int32_t input_channel, 
                   cmsis_nn_dims* filter_dims, int32_t kernel_size, 
                   cmsis_nn_dims* output_dims, int32_t output_size,  int32_t output_channel);

int32_t get_argmax_result(int8_t* output, int32_t output_size);







#endif