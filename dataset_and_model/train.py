import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

from model import Net


EPOCH = 10      # 轮次
BATCH_SIZE = 64 # 轮次大小


mnist_train = MNIST('./dataset_and_model/MNIST/data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST('./dataset_and_model/MNIST/data', train=False, download=True, transform=transforms.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True)


net = Net()
print(net)

optimizer = torch.optim.RMSprop(params=net.parameters(), lr=1e-4)   # 优化器
loss_func = nn.CrossEntropyLoss()   # 损失函数（含Softmax转化为概率分布）


losslist = []
acclist = []

for epoch in range(EPOCH):
    lossall = 0

    for x, t in loader_train:       # x  :[64,1,28,28]       t  :[64]
        out = net.forward(x)        # out:[64,10]   置信度，预测分数，logits（原始分数，未归一化）
        loss = loss_func(out, t)    # 损失计算
        lossall += loss.detach().cpu().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        acc = 0
        for x, t in loader_test:
            out = net(x)
            acc += sum(torch.argmax(out, dim=1) == t) # argmax, softmax 将预测分数转化为概率分布
        print('精确度：', acc.numpy()/len(mnist_test))

    print(lossall)
    losslist.append(lossall)
    acclist.append(acc.numpy()/len(mnist_test))

torch.save(net, './dataset_and_model/.net.plk')                      # 保存网络
torch.save(net.state_dict(), './dataset_and_model/.net_params.plk')  # 保存参数




plt.subplot(1,2,1)
plt.plot(list(range(EPOCH)), losslist, c='r')
plt.subplot(1,2,2)
plt.plot(list(range(EPOCH)), acclist, c='b')
plt.show()

