#ifndef __CMSIS_NN_HELPER_H
#define __CMSIS_NN_HELPER_H

#include "param.h"

void fc_set_context(fc_model_handle *fc_model);
void fc_set_dims(fc_model_handle *fc_model, int32_t input_dim, int32_t output_dim);
void fc_set_fc_params(fc_model_handle *fc_model, int32_t offset, int32_t min, int32_t max);
void fc_set_quant_params(fc_model_handle *fc_model, int32_t multiplier, int32_t shift);




#endif