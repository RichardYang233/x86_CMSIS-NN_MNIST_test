#include <stdio.h>
// #include "arm_math.h"
// #include "arm_nn_types.h"
#include "main.h"
#include "arm_nnfunctions.h"
#include "params_reader.h"
#include "NNInference.h"


int main(void) 
{
    int rows = 0;
    int cols = 0;

    char line[MAX_LINE_SIZE];
    FILE *file = open_csv(CSV_FILE_NAME);

    serch_lable_line(file, line, sizeof(line));
    parse_dim(line, &rows, &cols);

    float **params_array = allocate_2d_array(rows, cols); // 鍔ㄦ�佸垎閰嶄簩缁存暟缁?
    read_params(file, line, sizeof(line), params_array, rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            hidden_bias[j] = params_array[i][j];
        }
    }

    printf("%f\n", hidden_bias[0]);               // 偏置



    // 初始化参数
    init_nn_params();

    // 执行推理
    run_inference();

    // 输出分类结果
    // for (int i = 0; i < OUTPUT_SIZE; i++) {
    //     printf("Class %d score: %d\n", i, output[i]);
    // }


    printf("Run successfully !!!\n\n");

    return 0;
}



