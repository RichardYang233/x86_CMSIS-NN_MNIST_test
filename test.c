#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "params_reader.h"


// 全连接网络参数
#define INPUT_SIZE 784  // 输入尺寸（28x28 图像展平）
#define HIDDEN_SIZE 512 // 隐藏层节点数
#define OUTPUT_SIZE 10  // 输出类别数

// 模型参数读取相关
#define MAX_LINE_SIZE 100000
#define CSV_FILE_NAME "./FCNNModelCreater/params.csv"
#define LABEL "fc2.weight"


float drt_array[OUTPUT_SIZE];


int main()
{
    // 变量
    Dim_TypeDef Dim;
    char *label = LABEL;
    char line[MAX_LINE_SIZE];
    
    //
    FILE *file = open_csv(CSV_FILE_NAME);

    serch_lable_line(file, line, sizeof(line), label);
    parse_dim(line, &Dim);
    get_params(file, line, sizeof(line), drt_array, Dim);

    // 
    printf("Run successfully !!!\n\n");

    fclose(file);
    return 0;
}