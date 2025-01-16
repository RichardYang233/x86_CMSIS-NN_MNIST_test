# README

### 功能

1. 生成全连接神经网络
   1. 训练 MINST 数据集
   2. 保存 图像数据、模型参数 到 .csv文件

2. 量化图像数据、模型参数
3. 构建 FCNN 推理流程
4. 验证模型参数效果

### 文件

+ **dataset_and_model**
  + model.py：模型
  + train.py：训练并生辰参数
  + net_params_2_csv.py：将参数转为csv文件
+ DataReader：读取CSV参数
+ NNInference：构建 FCNN 推理流程
+ Quantification：量化工具