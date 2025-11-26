import os
import numpy as np
import torchvision
import time

# ---------------------------------------------------------
# 1. 加载数据
# ---------------------------------------------------------
def load_mnist_data(root='../data'):
    """
    使用 torchvision 下载并加载 MNIST 数据集
    然后把它们变成 numpy 数组，方便后面处理
    """
    print(f"正在加载数据，保存路径: {root} ...")
    os.makedirs(root, exist_ok=True)
    
    # 下载训练集 (60000张)
    train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=False)
    # 下载测试集 (10000张)
    test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=False)
    
    # 把数据转成 numpy 格式
    # X 是图片数据，y 是标签(0-9)
    X_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()
    
    print(f"训练集大小: {X_train.shape}") #  (60000, 28, 28)
    print(f"测试集大小: {X_test.shape}")   #  (10000, 28, 28)
    
    return X_train, y_train, X_test, y_test

# ---------------------------------------------------------
# 2. 数据预处理
# ---------------------------------------------------------
def preprocess_data(X_train, y_train, X_test, y_test):
    """
    对数据做三件事：
    1. 拉平：把 28x28 的图片变成 784 维的向量
    2. 打乱：把训练数据顺序打乱一下
    3. 中心化：把数据减去平均值（这是机器学习常见操作，为了让数据分布更好）
    """
    print("正在预处理数据...")
    
    # 1. 拉平 (Flatten)
    # .reshape(图片数量, -1) 会自动计算 -1 那个维度是多少 (这里是 28*28=784)
    # 转成 float32 类型，因为后面做减法可能是小数
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
    
    # 2. 打乱训练集 (Shuffle)
    # 生成一个随机的索引序列
    perm = np.random.permutation(X_train_flat.shape[0])
    X_train_flat = X_train_flat[perm]
    y_train = y_train[perm]
    
    # 3. 中心化 (Zero-mean)
    # 算出所有训练图片的平均脸 (784维)
    mean_image = np.mean(X_train_flat, axis=0)
    
    # 大家都减去这个平均脸
    X_train_centered = X_train_flat - mean_image
    X_test_centered = X_test_flat - mean_image
    
    print("数据预处理完成！")
    return X_train_centered, y_train, X_test_centered, y_test

# ---------------------------------------------------------
# 3. KNN 算法实现
# ---------------------------------------------------------
class KNN_Classifier:
    """
    自己动手实现的 KNN 分类器
    """
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """
        KNN 的训练其实就是把数据存起来，没别的
        """
        self.X_train = X
        self.y_train = y

    def compute_distances(self, X_test):
        """
        计算测试集里每张图，和训练集里所有图的距离
        
        这里用了一个数学公式来加速计算欧氏距离：(a-b)^2 = a^2 + b^2 - 2ab
        如果用两层 for 循环写，跑一次要好几分钟，太慢了。
        用矩阵运算可以几秒钟跑完。
        """
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        
        # 1. 算出测试集每个点的平方和
        test_sq = np.sum(X_test**2, axis=1, keepdims=True)
        
        # 2. 算出训练集每个点的平方和
        train_sq = np.sum(self.X_train**2, axis=1)
        
        # 3. 算出 2*a*b
        two_ab = 2 * np.dot(X_test, self.X_train.T)
        
        # 4. 拼起来：dist^2 = test^2 + train^2 - 2*test*train
        dists_sq = test_sq + train_sq - two_ab
        
        # 防止计算误差出现负数，把负数都变成0
        dists_sq = np.maximum(dists_sq, 0)
        
        return np.sqrt(dists_sq)

    def predict(self, dists, k=1):
        """
        根据算好的距离，选出最近的 k 个邻居，看看它们大多是什么数字
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test, dtype=int)
        
        for i in range(num_test):
            # 拿到第 i 张测试图 到 所有训练图 的距离
            current_dists = dists[i, :]
            
            # argsort 会返回从小到大排序后的 索引
            # 我们取前 k 个，这就是最近的 k 个邻居的索引
            closest_indices = np.argsort(current_dists)[:k]
            
            # 看看这 k 个邻居分别是什么数字
            closest_labels = self.y_train[closest_indices]
            
            # 投票：看看哪个数字出现次数最多
            # bincount 会统计每个数字出现了几次
            counts = np.bincount(closest_labels)
            
            # argmax 找出票数最多的那个数字
            y_pred[i] = np.argmax(counts)
            
        return y_pred

# ---------------------------------------------------------
# 4. 主程序
# ---------------------------------------------------------
if __name__ == "__main__":
    # 设置数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data')
    
    # 1. 加载数据
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_mnist_data(root=data_path)
    
    # 2. 预处理
    X_train, y_train, X_test, y_test = preprocess_data(X_train_raw, y_train_raw, X_test_raw, y_test_raw)
    
    # 3. 跑模型
    # 设置测试样本数量
    # 如果你想跑全部 10000 张，就把下面这行改成 num_test = 10000
    # 注意：跑全部数据可能需要几分钟时间（取决于电脑性能）
    num_test = 10000
    k_value = 3
    
    print(f"\n开始运行 KNN (k={k_value})，测试样本数: {num_test} ...")
    
    # 截取一部分测试数据
    X_test_subset = X_test[:num_test]
    y_test_subset = y_test[:num_test]
        
    # 实例化分类器
    knn = KNN_Classifier()
    knn.train(X_train, y_train)
    
    # 计时开始
    start = time.time()
    
    # 第一步：算距离
    print("正在计算距离矩阵...")
    dists = knn.compute_distances(X_test_subset)
    
    # 第二步：预测
    print("正在根据距离进行分类投票...")
    y_pred = knn.predict(dists, k=k_value)
    
    # 计时结束
    end = time.time()
    print(f"耗时: {end - start:.2f} 秒")
    
    # 4. 算算正确率
    # 预测对的数量
    correct_count = np.sum(y_pred == y_test_subset)
    accuracy = correct_count / num_test
    
    print(f"预测正确: {correct_count} / {num_test}")
    print(f"最终准确率: {accuracy:.2%}")
