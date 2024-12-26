#ifndef __TEST_DATASET_READER_H
#define __TEST_DATASET_READER_H

#define MAX_COLUMNS (28*28 + 1) 


int8_t get_image_label(int8_t image_data[]);
void get_single_image_data(char line[] ,int8_t image_data[]);




#endif



