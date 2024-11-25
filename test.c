#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "params_reader.h"
#include "main.h"




int main()
{
    FILE *file = open_csv(CSV_FILE_NAME);
    char line[MAX_LINE_SIZE];

    serch_lable_and_read_params(file, line);

    printf("Run successfully !!!\n\n");
    
    fclose(file);
    return 0;
}