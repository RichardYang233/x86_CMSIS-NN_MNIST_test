#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "params_reader.h"
#include "main.h"


int main()
{
    char line[MAX_LINE_SIZE];
    FILE *file = open_csv(CSV_FILE_NAME);


    // serch_lable_and_read_params(file, line, sizeof(line));
    int rows = 0;
    int cols = 0;

    serch_lable_line(file, line, sizeof(line));
    parse_dim(line, &rows, &cols);

    float drt_array[rows][cols];

    float **params_array = allocate_2d_array(rows, cols); // 动态分配二维数组
    read_params(file, line, sizeof(line), params_array, rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            drt_array[i][j] = params_array[i][j];
        }
    }

    free_2d_array(params_array, rows);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d: %f \n", j+1, drt_array[i][j]); 
        }
    }

    printf("Run successfully !!!\n\n");
    fclose(file);
    return 0;
}