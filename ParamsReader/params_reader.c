#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "stdint.h"

#include "main.h"



FILE *open_csv(const char *paramsFileName)
{
    FILE *file = fopen(paramsFileName, "r");
    return file;
}

int parse_dim(const char *line, int *rows, int *cols)
{
    // 查找维度部分 "torch.Size("
    const char *dim_start = strstr(line, "torch.Size([");
    if (dim_start == NULL)
    {
        return 0; // 未找到维度信息
    }

    if (sscanf(dim_start, "torch.Size([%d, %d])", rows, cols) == 2)
    {
        return 2;
    }
    else if (sscanf(dim_start, "torch.Size([%d])", cols) == 1)
    {
        *rows = 1;
        return 1;
    }
    else
    {
        return 0;
    }
}

// 判断是否识别到所选标签
bool is_lable(const char *line, char *lable)
{
    const char *lable_start = strstr(line, lable);
    if (lable_start == NULL)
    {
        return false;
    }
    else
    {
        return true;
    }
}

void read_params(FILE *file, char *line, int sizeOfline,float **array, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        if (fgets(line, sizeOfline, file) == NULL)
        {
            printf("Failed to read data for row %d.\n", i);
            break;
        }

        char *token = strtok(line, ","); // 分割当前行并解析列

        for (int j = 0; j < cols && token != NULL; j++)
        {
            array[i][j] = strtof(token, NULL); // 转换为浮点数
            //printf("%d: %f\n", (i + 1) * (j + 1), array[i][j]);
            token = strtok(NULL, ",");
        }
    }
    return;
}

serch_lable_and_read_params()
{
    int rows = 0;
    int cols = 0;
}

void copy(float **array, int8_t hidden_weights[HIDDEN_SIZE][INPUT_SIZE])
{
    float scale = 127.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++) 
    {
        for (int j = 0; j < INPUT_SIZE; j++) 
        {
            // 量化，将 float 转换为 int8_t
            float value = array[i][j];
            if (value > 1.0f) value = 1.0f; // 防止超出量化范围
            if (value < -1.0f) value = -1.0f;
            hidden_weights[i][j] = (int8_t)(value * scale);
        }
    }
}



// 动态分配二维数组
float **allocate_2d_array(int rows, int cols) 
{
    // 分配行指针数组
    float **array = (float **)malloc(rows * sizeof(float *));
    if (array == NULL) {
        printf("Memory allocation failed for row pointers!\n");
        return NULL;
    }

    // 分配每一行的列内存
    for (int i = 0; i < rows; i++) {
        array[i] = (float *)malloc(cols * sizeof(float));
        if (array[i] == NULL) {
            printf("Memory allocation failed for row %d!\n", i);

            // 释放已分配的内存以防止内存泄漏
            for (int j = 0; j < i; j++) {
                free(array[j]);
            }
            free(array);

            return NULL;
        }
    }

    return array;
}

void free_2d_array(float **array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]); // 释放每一行
    }
    free(array); // 释放行指针数组
}
