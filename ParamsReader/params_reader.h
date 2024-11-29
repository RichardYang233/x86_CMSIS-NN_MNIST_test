#ifndef __PARAMSREADER_H
#define __PARAMSREADER_H

#include <stdint.h>

#define MAX_LINE_SIZE 100000


typedef struct {
    int rows;
    int cols;
} Dim_TypeDef;


FILE *open_csv(const char *fliePath);
int parse_dim(const char *line, Dim_TypeDef *Dim);
bool is_label(const char *line, char *label);
void get_int8_params(FILE *file, char *line, int sizeOfline, int8_t array[], Dim_TypeDef Dim);
void get_int32_params(FILE *file, char *line, int sizeofline, int32_t array[], Dim_TypeDef Dim);

// void serch_lable_and_read_params(FILE *file, char *line, int sizeofline);
void serch_label_line(FILE* file, char *line, int sizeofline, char *label);


float **allocate_2d_array(int rows, int cols);
void free_2d_array(float **array, int rows);


#endif