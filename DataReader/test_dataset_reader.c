#include <stdio.h>
#include "stdint.h"

#include "test_dataset_reader.h"


#define TEST_DATASET_PATH "./NNInference/uantized_test_dataset.csv"
#define MAX_LINE_SIZE 100000



FILE *open_csv(const char *fliePath)
{
    FILE *file = fopen(fliePath, "r");
    return file;
}

int8_t get_single_image_data(FILE *file, char *line, int num)
{
    rewind(file);
    while (gets(line, ))
}

int8_t get_image_label(int8_t image_data[])
{
    return image_data[0];
}



int main(void)
{
    char line[MAX_LINE_SIZE];

    FILE *file = open_csv(TEST_DATASET_PATH);
    printf("%s", file);

    return 0;
}
