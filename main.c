#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// CMSIS-NN
#include "arm_nnfunctions.h"
// 相关
#include "main.h"
#include "tensor_utils.h"
#include "cmsis_nn_helper.h"
// 参数
#include "testset_image_array.h"
#include "testset_label_array.h"
#include "LeNet/conv1_kernel.h"
#include "LeNet/conv1_bias.h"
#include "LeNet/conv2_kernel.h"
#include "LeNet/conv2_bias.h"
#include "LeNet/fc1_weight.h"
#include "LeNet/fc1_bias.h"
#include "LeNet/fc2_weight.h"
#include "LeNet/fc2_bias.h"
#include "LeNet/fc3_weight.h"
#include "LeNet/fc3_bias.h"


/* 统计信息 */ 
int total = 0;
int correct = 0;
int cnt;

int main(void)
{   
    /* 所需配置 */ 
    cmsis_nn_conv_params conv_params;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_fc_params fc_params;

    cmsis_nn_per_channel_quant_params per_channel_quant_params;
    cmsis_nn_per_tensor_quant_params per_tensor_quant_params;

    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    /* 推理 */ 
    for(cnt = 0; cnt < DATASET_CNT; cnt++)
    {
        int8_t* input = testset_image_array[cnt];

        /* ----- conv1 ----- */ 
        int8_t output_conv1[24 * 24 * 6];

        int32_t multiplier_conv1[6];
        int32_t shift_conv1[6];
        CONV_SetQuant(&per_channel_quant_params, 6, multiplier_conv1, 1681627904, shift_conv1, -10);
        CONV_SetParams(&conv_params, 0, 0, 1, 0);
        CONV_SetDims(&input_dims, 28, 1, &filter_dims, 5, &output_dims, 24, 6);

        CONV_Run(&conv_params,
                 &per_channel_quant_params,
                 &input_dims,   input,
                 &filter_dims,  conv1_kernel,
                 &bias_dims,    conv1_bias,
                 NULL,
                 &output_dims,  output_conv1);
                          
        /* ----- relu1 ----- */
        arm_relu_q7(output_conv1, 24 * 24 * 6);

        /* ----- pool1 ----- */
        int8_t output_pool1[12 * 12 * 6];

        POOL_SetParams(&pool_params, 2, 0);
        POOL_SetDims(&input_dims, 24, 6, &filter_dims, 2, &output_dims, 12, 6);

        MAXPOOL_Run(&pool_params,
                    &input_dims,
                    output_conv1,
                    &filter_dims,
                    &output_dims,
                    output_pool1);

        /* ----- conv2 ----- */ 
        int8_t output_conv2[8 * 8 * 16];

        int32_t multiplier_conv2[16];
        int32_t shift_conv2[16];
        CONV_SetQuant(&per_channel_quant_params, 16, multiplier_conv2, 1963532932, shift_conv2, -9);
        CONV_SetParams(&conv_params, 0, 0, 1, 0);
        CONV_SetDims(&input_dims, 12, 6, &filter_dims, 5, &output_dims, 8, 16);

        CONV_Run(&conv_params,
                 &per_channel_quant_params,
                 &input_dims, output_pool1,  
                 &filter_dims, conv2_kernel,  
                 &bias_dims, conv2_bias,     
                 NULL,
                 &output_dims, output_conv2);
                    
        /* ----- relu2 ----- */
        arm_relu_q7(output_conv2, 8 * 8 * 16);

        /* ----- pool2 ----- */
        int8_t output_pool2[4 * 4 * 16];

        POOL_SetParams(&pool_params, 2, 0);
        POOL_SetDims(&input_dims, 8, 16, &filter_dims, 2, &output_dims, 4, 16);

        MAXPOOL_Run(&pool_params,
                    &input_dims,
                    output_conv2,
                    &filter_dims,
                    &output_dims,
                    output_pool2);

        /* ----- dim_convert ----- */
        convert_HWC_2_CHW(output_pool2, 4, 4, 16); // 将 CMSIS-NN 的卷积层输出到全连接层之前，需进行此操作

        /* ----- fc1 ----- */
        int8_t output_fc1[120];

        FC_SetParams(&fc_params, 0, 0, 0);
        FC_SetQuant(&per_tensor_quant_params, 2131751968, -10);
        FC_SetDims(&input_dims, 16 * 4 * 4, &filter_dims, &output_dims, 120);

        FC_Run(&fc_params, 
               &per_tensor_quant_params, 
               &input_dims,
               output_pool2,
               &filter_dims,
               fc1_weight,
               &bias_dims,
               fc1_bias,
               &output_dims,
               output_fc1);
                                        
        /* ----- relu3 ----- */
        arm_relu_q7(output_fc1, 120);

        /* ----- fc2 ----- */
        int8_t output_fc2[84];

        FC_SetParams(&fc_params, 0, 0, 0);
        FC_SetQuant(&per_tensor_quant_params, 1239864745, -8);
        FC_SetDims(&input_dims, 120, &filter_dims, &output_dims, 84);

        FC_Run(&fc_params, 
                &per_tensor_quant_params, 
                &input_dims,
                output_fc1,
                &filter_dims,
                fc2_weight,
                &bias_dims,
                fc2_bias,
                &output_dims,
                output_fc2);
        
        /* ----- relu4 ----- */
        arm_relu_q7(output_fc2, 84);

        /* ----- fc3 ----- */
        int8_t output_fc3[10];

        FC_SetParams(&fc_params, 0, 0, 0);
        FC_SetQuant(&per_tensor_quant_params, 1194481401, -9);
        FC_SetDims(&input_dims, 84, &filter_dims, &output_dims, 10);

        FC_Run(&fc_params, 
                &per_tensor_quant_params, 
                &input_dims,
                output_fc2,
                &filter_dims,
                fc3_weight,
                &bias_dims,
                fc3_bias,
                &output_dims,
                output_fc3);

        /* -------------- */
        /*     Argmax     */ 
        /* -------------- */
        int32_t result = get_argmax_result(output_fc3, 10);

        // 结果统计
        total++;
        if (result == testset_label_array[cnt])
        {
            correct++;
        }
    }

    printf("\nCMSIS-NN Accuracy: %.2f%%\n", (float)correct / (float)total * 100);
    return 0;
}