#include <stdio.h>
// #include "arm_math.h"
// #include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "main.h"
#include "params_reader.h"
#include "NNInference.h"
#include "test_dataset_reader.h"
#include "file_utils.h"
#include "result_evaluate.h"


// 模型参数读取相关
#define MAX_LINE_SIZE 100000
#define PARAMS_PATH "./NNInference/quantized_params.csv"
#define TEST_DATASET_PATH "./NNInference/quantized_test_dataset.csv"


int main(void) 
{
    // 变量
    Dim_TypeDef Dim;
    char line[MAX_LINE_SIZE];
    char *label;

    // 文件
    FILE *file = open_csv(PARAMS_PATH);
    FILE *file_image = open_csv(TEST_DATASET_PATH);

    /*----------------- 提取数据 ------------------*/

    // fc1.weight
    int8_t fc1_weight[INPUT_SIZE * HIDDEN_SIZE];
    label = "fc1.weight";
    serch_label_line(file, line, sizeof(line), label);
    find_dim(line, &Dim);
    get_int8_params(file, line, sizeof(line), fc1_weight, Dim);
    for(int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++)
    {
        hidden_weights[i] = fc1_weight[i];
    }

    // fc1.bias
    int32_t fc1_bias[HIDDEN_SIZE];
    label = "fc1.bias";
    serch_label_line(file, line, sizeof(line), label);
    find_dim(line, &Dim);
    get_int32_params(file, line, sizeof(line), fc1_bias, Dim);
    for(int i = 0; i < HIDDEN_SIZE; i++)
    {
        hidden_bias[i] = fc1_bias[i];
    }

    // fc2.weight
    int8_t fc2_weight[HIDDEN_SIZE * OUTPUT_SIZE];
    label = "fc2.weight";
    serch_label_line(file, line, sizeof(line), label);
    find_dim(line, &Dim);
    get_int8_params(file, line, sizeof(line), fc2_weight, Dim);
    for(int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++)
    {   
        output_weights[i] = fc2_weight[i];
    }

    // fc2.bias
    int32_t fc2_bias[OUTPUT_SIZE];
    label = "fc2.bias";
    serch_label_line(file, line, sizeof(line), label);
    find_dim(line, &Dim);
    get_int32_params(file, line, sizeof(line), fc2_bias, Dim);
    for(int i = 0; i < OUTPUT_SIZE; i++)
    {
        output_bias[i] = fc2_bias[i];
    }

    // input
    int8_t data[MAX_COLUMNS];
    char line_image[MAX_LINE_SIZE];

    int right_count = 0;
    int fault_count = 0;
    while (fgets(line_image, sizeof(line_image), file_image))
    {
        get_single_image_data(line_image, data);
        int label = get_image_label(data);

        for (int i = 1; i <= 28*28; i ++)
        {
            input[i] = data[i];
        }

        // 初始化参数
        init_nn_params();
        // 执行推理
        run_inference();

        // 获取推理结果
        int result = get_result(output);
        // 结果判断
        if (result == label)
        {
            right_count++;
        }
        else
        {
            fault_count++;
        }

    }

    printf("right_count: %d\n", right_count);
    printf("fault_count: %d\n", fault_count);
    printf("accuracy: %f\n", (float)right_count / (right_count + fault_count));

    // 输出分类结果
    // for (int i = 0; i < OUTPUT_SIZE; i++) {
    //     printf("Class %d score: %d\n", i, output[i]);
    // }

    printf("Run successfully !!!\n\n");

    fclose(file);
    fclose(file_image);
    return 0;
}



