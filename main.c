#include <stdio.h>
// CMSIS-NN
#include "arm_nnfunctions.h"
// 相关
#include "main.h"
#include "config.h"
#include "param.h"
#include "cmsis_nn_helper.h"
// 参数
#include "testset_image_array.h"
#include "testset_label_array.h"
#include "hidden_layer_weight.h"
#include "hidden_layer_bias.h"
#include "output_layer_weight.h"
#include "output_layer_bias.h"


int main(void)
{   
    int total = 0;
    int correct = 0;
    int cnt;

    for(cnt = 0; cnt < DATASET_CNT; cnt++)
    {
        /* -------------- */
        /*  hidden_layer  */ 
        /* -------------- */

        FC_set_context(&ctx);
        FC_set_dims(INPUT_SIZE, HIDDEN_SIZE);
        FC_set_fc_params(&fc_params, 0, -128, 127);
        FC_set_quant_params(&quant_params, 1468934400, -11);

        int8_t *input = testset_image_array[cnt];
        int8_t *weight = hidden_layer_weight;
        int32_t *bias = hidden_layer_bias;
        int8_t fc1_output[HIDDEN_SIZE];

        arm_cmsis_nn_status status = arm_fully_connected_s8(
            &ctx,
            &fc_params,
            &quant_params,
            &input_dims,
            input,
            &weight_dims,
            weight,
            &bias_dims,
            bias,
            &output_dims,
            fc1_output
        );
        if (status != ARM_CMSIS_NN_SUCCESS) {
            printf("Error: Fully connected layer computation failed.\n");
        }
        // 激活函数（ReLU）
        arm_relu_q7(fc1_output, sizeof(fc1_output)/sizeof(fc1_output[0]) );

        /* -------------- */
        /*  output_layer  */ 
        /* -------------- */

        FC_set_context(&ctx);
        FC_set_dims(HIDDEN_SIZE, OUTPUT_SIZE);
        FC_set_fc_params(&fc_params, 79, -128, 127);
        FC_set_quant_params(&quant_params, 1134667560, -10);

        input = fc1_output;
        weight = output_layer_weight;
        bias = output_layer_bias;
        int8_t fc2_output[10];

        status = arm_fully_connected_s8(
            &ctx,
            &fc_params,
            &quant_params,

            &input_dims,
            input,

            &weight_dims,
            weight,

            &bias_dims,
            bias,
            
            &output_dims,
            fc2_output
        );
        if (status != ARM_CMSIS_NN_SUCCESS) {
            printf("Error: Fully connected layer computation failed.\n");
        }
        // 激活函数（ReLU）
        arm_relu_q7(fc2_output, sizeof(fc2_output)/sizeof(fc2_output[0]));

        /* -------------- */
        /*     Argmax     */ 
        /* -------------- */

        int temp = 0;
        int result = 0;
        
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {   
            if (temp < fc2_output[i])
            {
                temp = fc2_output[i];
                result = i;
            }
        }
        
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