# README

### 功能概述

1. 生成全连接神经网络
   1. 训练 MINST 数据集
   2. 保存 图像数据、模型参数 到 .csv文件

2. 量化图像数据、模型参数
3. 构建 FCNN 推理流程
4. 验证模型参数效果

### 文件构成

+ DataReader：读取CSV参数
+ FCNNModelCreater：训练模型，提取模型参数
+ NNInference：构建 FCNN 推理流程
+ Quantification：量化工具