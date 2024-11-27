#include <stdio.h>
// #include "arm_math.h"
// #include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "main.h"
#include "params_reader.h"
#include "NNInference.h"


// 模型参数读取相关
#define MAX_LINE_SIZE 100000
#define CSV_FILE_NAME "./FCNNModelCreater/params.csv"
#define LABEL "fc1.bias"



int main(void) 
{
    // 变量
    Dim_TypeDef Dim;
    char line[MAX_LINE_SIZE];

    // 文件
    FILE *file = open_csv(CSV_FILE_NAME);

    // hidden_bias[]
    float array[HIDDEN_SIZE];
    char *label = "fc1.bias";
    serch_lable_line(file, line, sizeof(line), label);
    parse_dim(line, &Dim);
    get_bias_params(file, line, sizeof(line), array, Dim);

    // 找 scale
    float temp = 0;
    float drt = 0;
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        // temp = array[i];
        // if (temp < drt)
        // {
        //     drt = temp;
        // }
        printf("%d ", image[i]);
    }
   





    // 初始化参数
    //init_nn_params();

    // 执行推理
    //run_inference();

    // 输出分类结果
    // for (int i = 0; i < OUTPUT_SIZE; i++) {
    //     printf("Class %d score: %d\n", i, output[i]);
    // }


    printf("Run successfully !!!\n\n");

    fclose(file);
    return 0;
}



