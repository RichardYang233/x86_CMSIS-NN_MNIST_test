#ifndef __PARAMSREADER_H
#define __PARAMSREADER_H


#define MAX_LINE_SIZE 100000


typedef struct 
{
    int rows;
    int cols;
} Dim_TypeDef;


FILE *open_csv(const char *paramsFileName);
int parse_dim(const char *line, Dim_TypeDef *Dim);
bool is_lable(const char *line, char *lable);
void get_params(FILE *file, char *line, int sizeOfline, float array[], Dim_TypeDef Dim);


// void serch_lable_and_read_params(FILE *file, char *line, int sizeofline);
void serch_lable_line(FILE* file, char *line, int sizeofline, char *label);


float **allocate_2d_array(int rows, int cols);
void free_2d_array(float **array, int rows);


#endif