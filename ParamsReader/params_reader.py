import csv
import numpy as np

def parse_csv(file_path, label):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        
        for row in reader:
            if row[0] == label:  # 检查目标名称
                # 解析维度信息
                shape = eval(row[1].replace("torch.Size", ""))  # 转换字符串为元组
                
                # 开始读取数据部分
                data = []
                while True:
                    try:
                        data_row = next(reader)  # 读取下一行
                        # 将每行的数据转换为浮点数并追加到列表
                        data.extend([float(x) for x in data_row])
                        # 停止条件：数据行长度等于期望的总数据量
                        if len(data) == np.prod(shape):
                            break
                    except StopIteration:
                        raise ValueError("Unexpected end of file while reading data.")
                
                # 将数据重塑为目标形状
                data = np.array(data).reshape(shape)
                return shape, data

    raise ValueError(f"{label} not found in the CSV file.")


# 示例用法
file_path = ".\FCNNModelCreater\params.csv"
label = "fc1.weight"

try:
    shape, data = parse_csv(file_path, label)
    print(f"Shape: {shape}")
    print(f"Data:\n{data}")
except ValueError as e:
    print(e)




    
