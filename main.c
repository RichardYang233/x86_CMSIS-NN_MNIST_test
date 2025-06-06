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
    /* 统计信息 */ 
    int total = 0;
    int correct = 0;
    int cnt;
    
    /* 层定义 */ 
    fc_model_handle fc_hidden_layer;
    fc_set_context(&fc_hidden_layer);
    fc_set_dims(&fc_hidden_layer, INPUT_SIZE, HIDDEN_SIZE);
    fc_set_fc_params(&fc_hidden_layer, 0, -128, 127);
    fc_set_quant_params(&fc_hidden_layer, 1468934400, -11);

    fc_model_handle fc_output_layer;
    fc_set_context(&fc_output_layer);
    fc_set_dims(&fc_output_layer, HIDDEN_SIZE, OUTPUT_SIZE);
    fc_set_fc_params(&fc_output_layer, 79, -128, 127);
    fc_set_quant_params(&fc_output_layer, 1134667560, -10);

    /* 推理 */
    for(cnt = 0; cnt < DATASET_CNT; cnt++)
    {
        /* -------------- */
        /*  hidden_layer  */ 
        /* -------------- */
        int8_t *input = testset_image_array[cnt];
        int8_t *weight = hidden_layer_weight;
        int32_t *bias = hidden_layer_bias;
        int8_t fc1_output[HIDDEN_SIZE];

        arm_cmsis_nn_status status = arm_fully_connected_s8(
            &fc_hidden_layer.ctx,
            &fc_hidden_layer.fc_params,
            &fc_hidden_layer.quant_params,
            &fc_hidden_layer.input_dims,
            input,
            &fc_hidden_layer.weight_dims,
            weight,
            &fc_hidden_layer.bias_dims,
            bias,
            &fc_hidden_layer.output_dims,
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
        input = fc1_output;
        weight = output_layer_weight;
        bias = output_layer_bias;
        int8_t fc2_output[10];

        status = arm_fully_connected_s8(
            &fc_output_layer.ctx,
            &fc_output_layer.fc_params,
            &fc_output_layer.quant_params,
            &fc_output_layer.input_dims,
            input,
            &fc_output_layer.weight_dims,
            weight,
            &fc_output_layer.bias_dims,
            bias,
            &fc_output_layer.output_dims,
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