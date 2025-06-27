# README

### 简介

本仓库用于将量化参数通过 CMSIS-NN库，实现 C/C++ 部署。以检验 CMSIS-NN 量化模型 与 torch量化模型 是否对齐

### 依赖

+ C/C++
+ CMSIS-NN Libarary
+ makefile
+ [NNQuantParamExtractor ]()仓库

### 目录结构

+ [`CMSIS_NN`]()：CMSIS-NN Libarary，包含静态链接库
+ [`dataset`]()：储存 MNIST 数据
+ [`model`]()：储存模型参数
+ [`utils`]()
  + `cmsis_nn_helper`：对 CMSIS-NN Libarary 功能包装
  + `tensor_utils`：tensor 操作工具

+ `testset_2_array.py`：用于将测试集数据转为 C数组

### 如何使用

```makefile
make
```

