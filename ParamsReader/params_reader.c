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
    //  "torch.Size("
    const char *dim_start = strstr(line, "torch.Size([");
    if (dim_start == NULL)
    {
        return 0; // 
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

// 
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

        char *token = strtok(line, ","); // 

        for (int j = 0; j < cols && token != NULL; j++)
        {
            array[i][j] = strtof(token, NULL); // 
            //printf("%d: %f\n", (i + 1) * (j + 1), array[i][j]);
            token = strtok(NULL, ",");
        }
    }
    return;
}




void copy(float **array, int8_t hidden_weights[HIDDEN_SIZE][INPUT_SIZE])
{
    float scale = 127.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++) 
    {
        for (int j = 0; j < INPUT_SIZE; j++) 
        {
            //  float  int8_t
            float value = array[i][j];
            if (value > 1.0f) value = 1.0f; //
            if (value < -1.0f) value = -1.0f;
            hidden_weights[i][j] = (int8_t)(value * scale);
        }
    }
}



// 
float **allocate_2d_array(int rows, int cols) 
{
    // 
    float **array = (float **)malloc(rows * sizeof(float *));
    if (array == NULL) {
        printf("Memory allocation failed for row pointers!\n");
        return NULL;
    }

    // 
    for (int i = 0; i < rows; i++) {
        array[i] = (float *)malloc(cols * sizeof(float));
        if (array[i] == NULL) {
            printf("Memory allocation failed for row %d!\n", i);

            // 
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
        free(array[i]); // 
    }
    free(array); // 
}


void serch_lable_and_read_params(FILE *file, char *line)
{
    int rows = 0;
    int cols = 0;
    int current_line = 0;

    while (fgets(line, sizeof(line), file))
    {
        current_line++;
        line[strcspn(line, "\n")] = 0; // 移除换行符

        if (is_lable(line, LABLE) == false)
        {
            continue;
        }

        parse_dim(line, &rows, &cols);
        // float params_array[rows][cols];
        float **params_array = allocate_2d_array(rows, cols); // 动态分配二维数组
        
        // 读出数据
        read_params(file, line, sizeof(line), params_array, rows, cols);

        free_2d_array(params_array, rows);
        break;
        
    }

}