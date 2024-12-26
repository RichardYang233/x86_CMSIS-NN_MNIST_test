#include <stdio.h>
#include "file_utils.h"


FILE *open_csv(const char *fliePath)
{
    FILE *file = fopen(fliePath, "r");
    if (!file)
    {
        perror("Failed to open file");
        return NULL;
    }
    return file;
}