#ifndef __TEST_DATASET_READER_H
#define __TEST_DATASET_READER_H


int8_t get_image_label(int8_t image_data[]);
void get_single_image_data(char line[] ,int8_t image_data[]);
FILE *open_csv(const char *fliePath);




#endif



