#include <stdio.h>
#include <stdlib.h>
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
    /* 配置 */ 
    int32_t buf_size;
    cmsis_nn_context ctx;

    cmsis_nn_conv_params conv_params;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_fc_params fc_params;

    cmsis_nn_per_channel_quant_params per_channel_quant_params;
    cmsis_nn_per_tensor_quant_params per_tensor_quant_params;

    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    arm_cmsis_nn_status status;

    for(cnt = 0; cnt < DATASET_CNT; cnt++)
    {

        /* ----- conv1 ----- */ 
        int8_t output_conv1[24 * 24 * 6];
        int32_t multiplier_conv1[6] = {1681627904, 1681627904, 1681627904, 1681627904, 1681627904, 1681627904};
        int32_t shift_conv1[6] = {-10, -10, -10, -10, -10, -10};

        conv_params.activation.max = 127;
        conv_params.activation.min = -128;
        conv_params.input_offset = 0;
        conv_params.output_offset = 0;
        conv_params.stride.w = 1;
        conv_params.stride.h = 1;
        conv_params.padding.w = 0;
        conv_params.padding.h = 0;
        conv_params.dilation.w = 1;
        conv_params.dilation.h = 1;

        // conv_params = conv_set_params_s8(0, 0, 1, 0);

        per_channel_quant_params.multiplier = multiplier_conv1;
        per_channel_quant_params.shift = shift_conv1;

        input_dims.n = 1;
        input_dims.w = 28;
        input_dims.h = 28;
        input_dims.c = 1;

        // filter_dims.n = 6;
        filter_dims.w = 5;
        filter_dims.h = 5;
        filter_dims.c = 1;

        // bias_dims.n = 1;
        // bias_dims.w = 1;
        // bias_dims.h = 1;
        // bias_dims.c = 6;

        // output_dims.n = 1;
        output_dims.w = 24;
        output_dims.h = 24;
        output_dims.c = 6;

        buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
        ctx.buf = malloc(buf_size);
        ctx.size = buf_size;

        status = arm_convolve_s8(&ctx,
                                &conv_params,
                                &per_channel_quant_params,
                                &input_dims, 
                                testset_image_array[cnt], //
                                &filter_dims, 
                                conv1_kernel,  //
                                &bias_dims, 
                                conv1_bias,    //
                                NULL,
                                &output_dims, output_conv1);
        if (status != ARM_CMSIS_NN_SUCCESS){ 
            printf("Convolution failed, status = %d\n", status); 
        }

        if (ctx.buf){
            memset(ctx.buf, 0, buf_size);
            free(ctx.buf);
        }

        /* ----- relu1 ----- */
        arm_relu_q7(output_conv1, 24 * 24 * 6);

        /* ----- pool1 ----- */
        int8_t output_pool1[12 * 12 * 6];

        pool_params.activation.max = 127;
        pool_params.activation.min = -128;
        pool_params.padding.w = 0;
        pool_params.padding.h = 0;
        pool_params.stride.w = 2;
        pool_params.stride.h = 2;

        input_dims.n = 1;
        input_dims.w = 24;
        input_dims.h = 24;
        input_dims.c = 6;

        filter_dims.w = 2;
        filter_dims.h = 2;

        output_dims.w = 12;
        output_dims.h = 12;
        output_dims.c = 6;

        buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
        ctx.buf = malloc(buf_size);
        ctx.size = buf_size;

        status = arm_max_pool_s8(&ctx,
                                &pool_params,
                                &input_dims,
                                output_conv1,
                                &filter_dims,
                                &output_dims,
                                output_pool1);
        if (status != ARM_CMSIS_NN_SUCCESS){ 
            printf("Convolution failed, status = %d\n", status); 
        }
        if (ctx.buf){
            memset(ctx.buf, 0, buf_size);
            free(ctx.buf);
        }
        //print_CHW_from_HWC(output_pool1, 12, 12, 6);


        /* ----- conv2 ----- */ 

        int8_t output_conv2[8 * 8 * 16];
        int32_t multiplier_conv2[16] = {1963532932, 1963532932, 1963532932, 1963532932, 
                                    1963532932, 1963532932, 1963532932, 1963532932, 
                                    1963532932, 1963532932, 1963532932, 1963532932, 
                                    1963532932, 1963532932, 1963532932, 1963532932};
        int32_t shift_conv2[16] = {-9,-9,-9,-9,
                                -9,-9,-9,-9,
                                -9,-9,-9,-9,
                                -9,-9,-9,-9};

        conv_params.activation.max = 127;
        conv_params.activation.min = -128;
        conv_params.input_offset = 0;
        conv_params.output_offset = 0;
        conv_params.stride.w = 1;
        conv_params.stride.h = 1;
        conv_params.padding.w = 0;
        conv_params.padding.h = 0;
        conv_params.dilation.w = 1;
        conv_params.dilation.h = 1;

        per_channel_quant_params.multiplier = multiplier_conv2;
        per_channel_quant_params.shift = shift_conv2;

        input_dims.n = 1;
        input_dims.w = 12;
        input_dims.h = 12;
        input_dims.c = 6;

        // filter_dims.n = 16;
        filter_dims.w = 5;
        filter_dims.h = 5;
        filter_dims.c = 6;

        // bias_dims.n = 1;
        // bias_dims.w = 1;
        // bias_dims.h = 1;
        // bias_dims.c = 16;

        // output_dims.n = 1;
        output_dims.w = 8;
        output_dims.h = 8;
        output_dims.c = 16;

        buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
        ctx.buf = malloc(buf_size);
        ctx.size = buf_size;

        status = arm_convolve_s8(&ctx,
                                &conv_params,
                                &per_channel_quant_params,
                                &input_dims, output_pool1,  //
                                &filter_dims, conv2_kernel, // 
                                &bias_dims, conv2_bias,     //
                                NULL,
                                &output_dims, output_conv2);//
        if (status != ARM_CMSIS_NN_SUCCESS){
            printf("Convolution failed, status = %d\n", status); 
        }
        if (ctx.buf){
            memset(ctx.buf, 0, buf_size);
            free(ctx.buf);
        }

        /* ----- relu2 ----- */
        arm_relu_q7(output_conv2, 8 * 8 * 16);
        //print_CHW_from_HWC(output_conv2, 8, 8, 16);

        /* ----- pool2 ----- */
        int8_t output_pool2[4 * 4 * 16];

        pool_params.activation.max = 127;
        pool_params.activation.min = -128;
        pool_params.padding.w = 0;
        pool_params.padding.h = 0;
        pool_params.stride.w = 2;
        pool_params.stride.h = 2;

        input_dims.n = 1;
        input_dims.w = 8;
        input_dims.h = 8;
        input_dims.c = 16;

        filter_dims.w = 2;
        filter_dims.h = 2;

        output_dims.w = 4;
        output_dims.h = 4;
        output_dims.c = 16;

        buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
        ctx.buf = malloc(buf_size);
        ctx.size = buf_size;

        status = arm_max_pool_s8(&ctx,
                                &pool_params,
                                &input_dims,
                                output_conv2,
                                &filter_dims,
                                &output_dims,
                                output_pool2);
        if (status != ARM_CMSIS_NN_SUCCESS){ 
            printf("Convolution failed, status = %d\n", status); 
        }
        if (ctx.buf){
            memset(ctx.buf, 0, buf_size);
            free(ctx.buf);
        }
        // print_CHW_from_HWC(output_pool2, 4, 4, 16);

        /* ----- fc1 ----- */
        convert_HWC_2_CHW(output_pool2, 4, 4, 16); // 将 CMSIS-NN 的卷积层输出到全连接层之前，需进行此操作
        int8_t output_fc1[120];
        
        ctx.buf = NULL;
        ctx.size = 0;

        fc_params.activation.max = 127;
        fc_params.activation.min = -128;
        fc_params.input_offset = 0;
        fc_params.filter_offset = 0;
        fc_params.output_offset = 0;
        
        per_tensor_quant_params.multiplier = 2131751968;
        per_tensor_quant_params.shift = -10;

        input_dims.n = 1;
        input_dims.w = 1;
        input_dims.h = 1;
        input_dims.c = 16 * 4 * 4;

        filter_dims.n = 16 * 4 * 4;
        filter_dims.w = 1;
        filter_dims.h = 1;
        filter_dims.c = 120;

        bias_dims.n = 1;
        bias_dims.w = 1;
        bias_dims.h = 1;
        bias_dims.c = 120;

        output_dims.n = 1;
        output_dims.w = 1;
        output_dims.h = 1;
        output_dims.c = 120;

        status = arm_fully_connected_s8(&ctx,
                                        &fc_params, 
                                        &per_tensor_quant_params, 
                                        &input_dims,
                                        output_pool2,
                                        &filter_dims,
                                        fc1_weight,
                                        &bias_dims,
                                        fc1_bias,
                                        &output_dims,
                                        output_fc1);
        if (status != ARM_CMSIS_NN_SUCCESS){ 
            printf("Convolution failed, status = %d\n", status); 
        }
        /* ----- relu3 ----- */
        arm_relu_q7(output_fc1, 120);

        /* ----- fc2 ----- */
        int8_t output_fc2[84];
        
        ctx.buf = NULL;
        ctx.size = 0;

        fc_params.activation.max = 127;
        fc_params.activation.min = -128;
        fc_params.input_offset = 0;
        fc_params.filter_offset = 0;
        fc_params.output_offset = 0;
        
        per_tensor_quant_params.multiplier = 1239864745;
        per_tensor_quant_params.shift = -8;

        input_dims.n = 1;
        input_dims.w = 1;
        input_dims.h = 1;
        input_dims.c = 120;

        filter_dims.n = 120;
        filter_dims.w = 1;
        filter_dims.h = 1;
        filter_dims.c = 84;

        bias_dims.n = 1;
        bias_dims.w = 1;
        bias_dims.h = 1;
        bias_dims.c = 84;

        output_dims.n = 1;
        output_dims.w = 1;
        output_dims.h = 1;
        output_dims.c = 84;

        status = arm_fully_connected_s8(&ctx,
                                        &fc_params, 
                                        &per_tensor_quant_params, 
                                        &input_dims,
                                        output_fc1,
                                        &filter_dims,
                                        fc2_weight,
                                        &bias_dims,
                                        fc2_bias,
                                        &output_dims,
                                        output_fc2);
        if (status != ARM_CMSIS_NN_SUCCESS){ 
            printf("Convolution failed, status = %d\n", status); 
        }
        /* ----- relu4 ----- */
        arm_relu_q7(output_fc2, 84);

        /* ----- fc2 ----- */
        int8_t output_fc3[10];
        
        ctx.buf = NULL;
        ctx.size = 0;

        fc_params.activation.max = 127;
        fc_params.activation.min = -128;
        fc_params.input_offset = 0;
        fc_params.filter_offset = 0;
        fc_params.output_offset = 62;
        
        per_tensor_quant_params.multiplier = 1194481401;
        per_tensor_quant_params.shift = -9;

        input_dims.n = 1;
        input_dims.w = 1;
        input_dims.h = 1;
        input_dims.c = 84;

        filter_dims.n = 84;
        filter_dims.w = 1;
        filter_dims.h = 1;
        filter_dims.c = 10;

        bias_dims.n = 1;
        bias_dims.w = 1;
        bias_dims.h = 1;
        bias_dims.c = 10;

        output_dims.n = 1;
        output_dims.w = 1;
        output_dims.h = 1;
        output_dims.c = 10;

        status = arm_fully_connected_s8(&ctx,
                                        &fc_params, 
                                        &per_tensor_quant_params, 
                                        &input_dims,
                                        output_fc2,
                                        &filter_dims,
                                        fc3_weight,
                                        &bias_dims,
                                        fc3_bias,
                                        &output_dims,
                                        output_fc3);
        if (status != ARM_CMSIS_NN_SUCCESS){ 
            printf("Convolution failed, status = %d\n", status); 
        }
        /* ----- relu5 ----- */
        arm_relu_q7(output_fc3, 10);

        /* -------------- */
        /*     Argmax     */ 
        /* -------------- */
        int temp = 0;
        int result = 0;
        
        for (int i = 0; i < 10; i++)
        {   
            if (temp < output_fc3[i])
            {
                temp = output_fc3[i];
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