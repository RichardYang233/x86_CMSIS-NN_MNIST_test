#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "stdint.h"

#include "main.h"
#include "params_reader.h"



// 
bool is_label(const char *line, char *label)
{
    const char *label_start = strstr(line, label);
    if (label_start == NULL)
    {
        return false;
    }
    else
    {
        return true;
    }
}

// 
int find_dim(const char *line, Dim_TypeDef *Dim)
{
    //  "torch.Size("
    const char *dim_start = strstr(line, "torch.Size([");
    if (dim_start == NULL)
    {
        return 0; // 
    }

    if (sscanf(dim_start, "torch.Size([%d, %d])", &Dim->rows, &Dim->cols) == 2)
    {
        return 2;
    }
    else if (sscanf(dim_start, "torch.Size([%d])", &Dim->cols) == 1)
    {
        Dim->rows = 1;
        return 1;
    }
    else
    {
        return 0;
    }
}

//
void get_int8_params(FILE *file, char *line, int sizeofline, int8_t array[], Dim_TypeDef Dim)
{
    // printf("%d, %d\n", Dim.rows, Dim.cols); // test
    for (int i = 0; i < Dim.rows; i++)
    {
        if (fgets(line, sizeofline, file) == NULL)
        {
            printf("Failed to read data for row %d.\n", i);
            break;
        }

        char *token = strtok(line, ","); // 

        for (int j = 0; j < Dim.cols && token != NULL; j++)
        {
            array[i*Dim.cols + j] = strtof(token, NULL); // 
            // printf("%d: %f\n", i*Dim.cols+j+1, array[i*Dim.cols+j]); // test
            token = strtok(NULL, ",");
        }
    }
    return;
}

void get_int32_params(FILE *file, char *line, int sizeofline, int32_t array[], Dim_TypeDef Dim)
{
    // printf("%d, %d\n", Dim.rows, Dim.cols); // test
    for (int i = 0; i < Dim.rows; i++)
    {
        if (fgets(line, sizeofline, file) == NULL)
        {
            printf("Failed to read data for row %d.\n", i);
            break;
        }

        char *token = strtok(line, ","); // 

        for (int j = 0; j < Dim.cols && token != NULL; j++)
        {
            array[i*Dim.cols + j] = strtof(token, NULL); // 
            // printf("%d: %f\n", i*Dim.cols+j+1, array[i*Dim.cols+j]); // test
            token = strtok(NULL, ",");
        }
    }
    return;
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

// 
void free_2d_array(float **array, int rows) 
{
    for (int i = 0; i < rows; i++) {
        free(array[i]); // 
    }
    free(array); // 
}

// 
void serch_label_line(FILE* file, char *line, int sizeofline, char *label)
{
    rewind(file);
    while (fgets(line, sizeofline, file))
    {
        line[strcspn(line, "\n")] = 0; // 去除换行�?
        if (is_label(line, label) == true)
        {
            break;
        }
    }
}
