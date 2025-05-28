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

    for(cnt = 0; cnt < 10000; cnt++)
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

        // 激活函数（ReLU）
        for (int i = 0; i < HIDDEN_SIZE; i++) 
        {
            fc1_output[i] = fc1_output[i] < 0 ? 0 : fc1_output[i];
        }


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

        // 激活函数（ReLU）
        for (int i = 0; i < OUTPUT_SIZE; i++) 
        {
            fc2_output[i] = fc2_output[i] < 0 ? 0 : fc2_output[i];
        }

        /* -------------- */
        /*     Argmax     */ 
        /* -------------- */
        int temp = 0;
        int result = 0;
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {   
            if (temp <= fc2_output[i])
            {
                temp = fc2_output[i];
                result = i;
            }
        }

        total++;
        if (result == testset_label_array[cnt])
        {
            correct++;
        }
    }

    printf("%d\n%d\n", correct, total);

    return 0;
}