import csv
import numpy as np
import shutil



class CSVHandler:
    def __init__(self, src_path, drt_path):
        self.src_path  = src_path
        self.file_path = drt_path
        self.label = None
        self.copy_csv()
    
    def copy_csv(self): 
        shutil.copy(self.src_path, self.file_path)
        print('copy successful !!!')

    def set_label(self, label: str):
        self.label = label

        
    def read_params(self):
        with open(self.file_path, mode='r') as file:
            reader = csv.reader(file)
            # 逐行检查
            for row in reader:
                if row[0] == self.label:  # 比对目标名称
                    # 解析维度信息
                    shape = eval(row[1].replace("torch.Size", "")) # 转换字符串为元组
                    # 读取数据部分
                    data = []
                    while True:
                        try:
                            data_row = next(reader)  # 读取下一行
                            data.extend([float(x) for x in data_row]) # 将每行的数据转换为浮点数并追加到列表
                            # 停止条件：数据行长度等于期望的总数据量
                            if len(data) == np.prod(shape):
                                break
                        except StopIteration:
                            raise ValueError("Unexpected end of file while reading data.")
                    # 将数据重塑为目标形状
                    data = np.array(data).reshape(shape)
                    return data # return shape
        raise ValueError(f"{self.label} not found in the CSV file.")


    def output_quantized_params(self, quantized_data):
        with open(self.file_path, mode='r' , newline='') as file:
            reader = csv.reader(file)
            data = list(reader)

        # 查找目标行索引
        target_index = None
        for i, row in enumerate(data):
            if self.label in row:
                target_index = i
                break
        
        # 替换数据
        if target_index is not None:
            for i in range(quantized_data.shape[0]):
                row = quantized_data[i, :].tolist()
                data[target_index + 1 + i] = row # 插入新数据行

            with open(self.file_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data)
        else:
            print("没有找到目标行 ！！！")


# file_path = "./FCNNModelCreater/params.csv"
# label = "fc1.weight"

# try:
#     shape, data = parse_csv(file_path, label)
#     print(f"Shape: {shape}")
#     print(f"Data:\n{data}")
# except ValueError as e:
#     print(e)




    
