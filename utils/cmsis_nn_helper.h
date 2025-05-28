#ifndef __CMSIS_NN_HELPER_H
#define __CMSIS_NN_HELPER_H

void FC_set_context(cmsis_nn_context *ctx);
void FC_set_dims(int32_t input_dim, int32_t output_dim);
void FC_set_fc_params(cmsis_nn_fc_params *fc_params, int32_t offset, int32_t min, int32_t max);
void FC_set_quant_params(cmsis_nn_per_tensor_quant_params *quant_params, int32_t multiplier, int32_t shift);

// void set_fc_params();

// void set_quant_params();




#endif