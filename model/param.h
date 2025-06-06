#ifndef __PARAM_H
#define __PARAM_H

#include "arm_nnfunctions.h"
#include "config.h"

/* ----------------------------------------------------------- */
/* 模型参数                                                     */
/* ----------------------------------------------------------- */

typedef struct {
    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims, weight_dims, bias_dims, output_dims;
} fc_model_handle;


#endif