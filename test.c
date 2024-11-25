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
    
    serch_lable_and_read_params(file, line, sizeof(line));

    printf("Run successfully !!!\n\n");
    
    fclose(file);
    return 0;
}