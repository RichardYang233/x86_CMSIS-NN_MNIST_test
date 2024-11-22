#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "params_reader.h"
#include "main.h"



int main()
{
    FILE *file = open_csv(CSV_FILE_NAME);

    // 初始化参�?
    int rows = 0;
    int cols = 0;
    char line[MAX_LINE_SIZE];
    int current_line = 0;

    // 逐行读取
    while (fgets(line, sizeof(line), file))
    {
        current_line++;
        line[strcspn(line, "\n")] = 0; // 移除换行�?

        if (is_lable(line, LABLE) == false)
        {
            continue;
        }

        parse_dim(line, &rows, &cols);
        //float params_array[rows][cols];
        float **params_array = allocate_2d_array(rows, cols); // 动态分配二维数�?
        
        // 读出数据
        read_params(file, line, sizeof(line), params_array, rows, cols);

        free_2d_array(params_array, rows);
        break;
        
    }

    printf("Run successfully !!!\n\n");
    
    fclose(file);
    return 0;
}