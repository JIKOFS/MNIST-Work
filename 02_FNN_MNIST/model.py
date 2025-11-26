import torch
import torch.nn as nn

class SimpleFNN(nn.Module):
    def __init__(self):
        super(SimpleFNN, self).__init__()
        
        # 定义一个简单的 3 层神经网络
        # 输入层: 784 个节点 (对应 28x28 像素)
        # 隐藏层1: 512 个节点
        # 隐藏层2: 256 个节点
        # 输出层: 10 个节点 (对应数字 0-9)
        
        self.layer1 = nn.Linear(784, 512) # 第一层线性变换
        self.relu1 = nn.ReLU()            # 激活函数 (把负数变成0)
        
        self.layer2 = nn.Linear(512, 256) # 第二层
        self.relu2 = nn.ReLU()
        
        self.layer3 = nn.Linear(256, 10)  # 输出层
        
    def forward(self, x):
        # 前向传播：数据流过网络的路径
        
        # x 的形状可能是 (64, 1, 28, 28)，我们需要把它拉平变成 (64, 784)
        # x.size(0) 是 batch_size (一次输入的图片数量)
        x = x.view(x.size(0), -1)
        
        x = self.layer1(x)
        x = self.relu1(x)
        
        x = self.layer2(x)
        x = self.relu2(x)
        
        x = self.layer3(x)
        return x
