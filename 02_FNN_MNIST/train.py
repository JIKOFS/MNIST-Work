import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
from model import SimpleFNN

def main():
    # ---------------------------------------------------------
    # 1. 基础设置
    # ---------------------------------------------------------
    # 每次训练 64 张图片
    BATCH_SIZE = 64
    # 学习率 (控制模型学得有多快)
    LEARNING_RATE = 0.001
    # 训练轮数 (把所有数据看几遍)
    EPOCHS = 5
    
    # 检查有没有显卡，有就用显卡跑，没有就用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    # ---------------------------------------------------------
    # 2. 准备数据
    # ---------------------------------------------------------
    # 定义数据的转换操作：变成 Tensor 格式 -> 归一化 (让数据在 0 附近)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 的均值和方差
    ])
    
    # 数据存放路径
    data_path = os.path.join('..', 'data')
    
    print("正在加载数据...")
    # 下载/加载训练集和测试集
    train_data = torchvision.datasets.MNIST(root=data_path, train=True, 
                                          transform=transform, download=False)
    test_data = torchvision.datasets.MNIST(root=data_path, train=False, 
                                         transform=transform, download=False)

    # 弄成加载器，方便一批一批地拿数据
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # ---------------------------------------------------------
    # 3. 创建模型
    # ---------------------------------------------------------
    model = SimpleFNN().to(device)
    
    # 损失函数：交叉熵损失 (分类任务专用)
    criterion = nn.CrossEntropyLoss()
    # 优化器：Adam (一种梯度下降算法)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---------------------------------------------------------
    # 4. 开始训练
    # ---------------------------------------------------------
    print("\n开始训练模型...")
    total_steps = len(train_loader)
    
    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            # 把数据搬到显卡/CPU上
            images = images.to(device)
            labels = labels.to(device)
            
            # 1. 前向传播：算预测结果
            outputs = model(images)
            loss = criterion(outputs, labels) # 算算错得有多离谱
            
            # 2. 反向传播：更新参数
            optimizer.zero_grad() # 清空上一步的梯度
            loss.backward()       # 算梯度
            optimizer.step()      # 更新参数
            
            # 每 100 步打印一次进度
            if (i+1) % 100 == 0:
                print(f'第 {epoch+1} 轮, 进度 [{i+1}/{total_steps}], 当前误差(Loss): {loss.item():.4f}')

    # ---------------------------------------------------------
    # 5. 测试效果
    # ---------------------------------------------------------
    print("\n正在测试模型准确率...")
    model.eval() # 切换到测试模式
    correct = 0
    total = 0
    
    # 测试的时候不需要算梯度，省内存
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            # 选分数最高的那个作为预测结果
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'最终测试准确率: {100 * correct / total:.2f}%')

    # ---------------------------------------------------------
    # 6. 保存模型
    # ---------------------------------------------------------
    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'mnist_fnn.pth')
    
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")

if __name__ == "__main__":
    main()
