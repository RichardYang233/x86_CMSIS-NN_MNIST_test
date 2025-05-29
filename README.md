# README

### 简介

本仓库用于将量化参数通过 CMSIS-NN库，实现 C/C++ 部署。以检验 CMSIS-NN量化模型 与 torch量化模型 是否对齐

量化参数获取： [NNQuantParamExtractor ]()仓库

### 文件构成

| 文件夹     | 描述           | 备注 |
| ---------- | -------------- | ---- |
| `CMSIS_NN` | CMSIS-NN 库    |      |
| `dataset`  | mnist 数据集   |      |
| `model`    | 模型信息和参数 |      |
| `utils`    |                |      |

`main.c`：

`testset_2_array.py`：用于将测试集数据转为 C数组



---
