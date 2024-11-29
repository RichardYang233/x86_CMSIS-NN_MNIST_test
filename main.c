#include <stdio.h>
// #include "arm_math.h"
// #include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "main.h"
#include "params_reader.h"
#include "NNInference.h"


// 模型参数读取相关
#define MAX_LINE_SIZE 100000
#define CSV_FILE_PATH "./NNInference/quantized_params.csv"


int main(void) 
{
    // 变量
    Dim_TypeDef Dim;
    char line[MAX_LINE_SIZE];
    char *label;

    // 文件
    FILE *file = open_csv(CSV_FILE_PATH);

    /*----------------- 提取数据 ------------------*/

    // input


    // fc1.weight
    int8_t fc1_weight[INPUT_SIZE * HIDDEN_SIZE];
    label = "fc1.weight";
    serch_label_line(file, line, sizeof(line), label);
    parse_dim(line, &Dim);
    get_int8_params(file, line, sizeof(line), fc1_weight, Dim);
    for(int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++)
    {
        hidden_weights[i] = fc1_weight[i];
    }

    // fc1.bias
    int32_t fc1_bias[HIDDEN_SIZE];
    label = "fc1.bias";
    serch_label_line(file, line, sizeof(line), label);
    parse_dim(line, &Dim);
    get_int32_params(file, line, sizeof(line), fc1_bias, Dim);
    for(int i = 0; i < HIDDEN_SIZE; i++)
    {
        hidden_bias[i] = fc1_bias[i];
    }

    // fc2.weight
    int8_t fc2_weight[HIDDEN_SIZE * OUTPUT_SIZE];
    label = "fc2.weight";
    serch_label_line(file, line, sizeof(line), label);
    parse_dim(line, &Dim);
    get_int8_params(file, line, sizeof(line), fc2_weight, Dim);
    for(int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++)
    {   
        output_weights[i] = fc2_weight[i];
    }

    // fc2.bias
    int32_t fc2_bias[OUTPUT_SIZE];
    label = "fc2.bias";
    serch_label_line(file, line, sizeof(line), label);
    parse_dim(line, &Dim);
    get_int32_params(file, line, sizeof(line), fc2_bias, Dim);
    for(int i = 0; i < OUTPUT_SIZE; i++)
    {
        output_bias[i] = fc2_bias[i];
    }

    
    // 初始化参数
    init_nn_params();

    // 执行推理
    run_inference();

    // 输出分类结果
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("Class %d score: %d\n", i, output[i]);
    }


    printf("Run successfully !!!\n\n");

    fclose(file);
    return 0;
}



