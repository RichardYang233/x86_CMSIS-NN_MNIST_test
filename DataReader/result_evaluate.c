#include <stdio.h>
#include "stdint.h"


int get_result(int8_t output[])
{
    int temp = -128;
    int result;
    for(int i = 0; i < 10; i ++)
    {
        if (output[i] > temp) 
        {
            temp = output[i];
            result = i;
        }
    }
    return result;
}