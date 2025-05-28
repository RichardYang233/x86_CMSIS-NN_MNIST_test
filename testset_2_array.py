import torch
import numpy as np
from torchvision import datasets, transforms


# -------- config ----------# 

scale = torch.tensor(0.00787402) # 输入值 scale，量化到 [0, 127]

test_dataset = datasets.MNIST(root='./dataset/mnist', train=False, download=True, transform=transforms.ToTensor())

rows = len(test_dataset)
cols = 28 * 28

# -------- image ----------# 

file = "dataset"
name = "testset_image_array"

with open(f"./{file}./{name}.h", "w") as f:
        # 头文件
        f.write(f"#ifndef {name.upper()}_H\n#define {name.upper()}_H\n\n")
        f.write(f"#include <stdint.h>\n\n")

        # 尺寸定义
        f.write(f"#define {name.upper()}_ROWS {rows}\n")
        f.write(f"#define {name.upper()}_COLS {cols}\n\n")

        # 数组
        f.write(f"const int8_t {name}[{name.upper()}_ROWS][{name.upper()}_COLS] = {{\n")

        for i in range(rows):
        # for i in range(20):    # 生成较少数据以供测试
            image, _ = test_dataset[i]
            image_q = torch.round(image.view(-1) / scale).to(torch.int8)
            image_np = image_q.numpy()

            row_str = ", ".join(str(x) for x in image_np)
            f.write(f"    {{{row_str}}},\n")
                  
        f.write("};\n\n#endif\n")

# -------- label ----------# 

# -------- label ----------# 

file = "dataset"
name = "testset_label_array"

# 获取标签数组（如果没提前提取的话）
labels = test_dataset.targets if hasattr(test_dataset, 'targets') else [test_dataset[i][1] for i in range(len(test_dataset))]

with open(f"./{file}/{name}.h", "w") as f:
    # 头文件保护
    f.write(f"#ifndef {name.upper()}_H\n#define {name.upper()}_H\n\n")
    f.write(f"#include <stdint.h>\n\n")

    # 尺寸定义
    length = len(labels)
    f.write(f"#define {name.upper()}_LEN {length}\n\n")

    # 数组定义
    f.write(f"const uint8_t {name}[{name.upper()}_LEN] = {{\n")

    # 写入标签
    for i, label in enumerate(labels):
        if i % 20 == 0:
            f.write("    ")  # 每 20 个换一行，更美观
        f.write(f"{int(label)}, ")
        if (i + 1) % 20 == 0:
            f.write("\n")

    f.write("\n};\n\n#endif\n")

