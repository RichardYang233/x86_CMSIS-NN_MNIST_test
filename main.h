#ifndef __MAIN_H
#define __MAIN_H

// 全连接网络参数
#define INPUT_SIZE 784  // 输入尺寸（28x28 图像展平）
#define HIDDEN_SIZE 512 // 隐藏层节点数
#define OUTPUT_SIZE 10  // 输出类别数

// 模型参数读取相关
#define MAX_LINE_SIZE 100000
#define CSV_FILE_NAME "./FCNNModelCreater/params.csv"
#define LABEL "fc1.bias"


#endif