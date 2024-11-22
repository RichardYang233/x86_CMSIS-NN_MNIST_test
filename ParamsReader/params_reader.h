#ifndef __PARAMSREADER_H
#define __PARAMSREADER_H


#define MAX_LINE_SIZE 100000


FILE *open_csv(const char *paramsFileName);
int parse_dim(const char *line, int *rows, int *cols);
bool is_lable(const char *line, char *lable);
void read_params(FILE *file, char *line, int sizeOfline,float **array, int rows, int cols);
float **allocate_2d_array(int rows, int cols);
void free_2d_array(float **array, int rows);


#endif