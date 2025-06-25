#ifndef __CONFIG_H
#define __CONFIG_H

/* ----------------------------------------------------------- */
/* 量化参数                                                     */ 
/* ----------------------------------------------------------- */


/* ----------------------------------------------------------- */
/* 模型结构                                                     */ 
/* ----------------------------------------------------------- */

/* FCNet */
#define INPUT_SIZE          784  // 28 x 28
#define HIDDEN_SIZE         512 
#define OUTPUT_SIZE         10

/* LeNet */
#define CONV1_INPUT_LEN     28
#define CONV1_INPUT_DIM     1
#define CONV1_KERNEL_LEN    5
#define CONV1_KERNEL_DIM    6
#define CONV1_OUTPUT_LEN    (CONV1_INPUT_LEN - CONV1_KERNEL_LEN + 1)    // stride 不为 1 时 需要考虑进来
#define CONV1_OUTPUT_DIM    CONV1_KERNEL_DIM
#define CONV1_STRIDE        1   // h * w
#define CONV1_PADDING       0   // h * w
#define CONV1_DILATION      1   // h * w

#define POOL1_INPUT_LEN     CONV1_OUTPUT_LEN
#define POOL1_INPUT_DIM     CONV1_OUTPUT_DIM
#define POOL1_KERNEL_LEN    2
#define POOL1_KERNEL_DIM    CONV1_OUTPUT_DIM
#define POOL1_OUPUT_LEN     12
#define POOL1_OUPUT_DIM     CONV1_OUTPUT_DIM
#define POOL1_STRIDE        2
#define POOL1_PADDING       0






#endif