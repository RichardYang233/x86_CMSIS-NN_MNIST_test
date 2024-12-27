#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stdint.h"

#include "test_dataset_reader.h"




void get_single_image_data(char line[] ,int8_t image_data[])
{
    line[strcspn(line, "\n")] = '\0';

    int column = 0;
    char *token = strtok(line, ",");

    while (token != NULL)
    {
        image_data[column] = atoi(token);
        column ++;
        token = strtok(NULL, ",");
    }
    // return column;
}

int8_t get_image_label(int8_t image_data[])
{
    return image_data[0];
}



/* ===== test ===== */
// int main(void)
// {
//     int8_t data[MAX_COLUMNS];
//     char line[MAX_LINE_SIZE];

//     FILE *file = open_csv(TEST_DATASET_PATH);
//     while (fgets(line, sizeof(line), file))
//     {
//         get_single_image_data(line, data);
//         int label = get_image_label(data);
//         /* ===== test ===== */
//         // printf("Row data: ");
//         // for (int i = 0; i < 10; i++) {
//         //     printf("%d ", data[i]);
//         // }
//         // printf("\n");  
//     }

//     return 0;
// }
