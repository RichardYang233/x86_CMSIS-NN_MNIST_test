from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # 扁平化
        self.hidden_layer = nn.Linear(28 * 28, 512) # 隐藏层
        self.relu = nn.ReLU()   # 激活函数
        self.output_layer = nn.Linear(512, 10)  # 输出层

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

