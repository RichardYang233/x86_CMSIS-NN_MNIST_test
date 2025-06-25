#ifndef __CMSIS_NN_HELPER_H
#define __CMSIS_NN_HELPER_H

#include "param.h"





// ----------------------------------------------------------------------
// 全连接层
// ----------------------------------------------------------------------

void fc_set_context(fc_model_handle *fc_model);
void fc_set_dims(fc_model_handle *fc_model, int32_t input_dim, int32_t output_dim);
void fc_set_fc_params(fc_model_handle *fc_model, int32_t offset, int32_t min, int32_t max);
void fc_set_quant_params(fc_model_handle *fc_model, int32_t multiplier, int32_t shift);

cmsis_nn_conv_params conv_set_params_s8(int32_t input_offset, int32_t output_offset, int32_t stride, int32_t padding);






#endif