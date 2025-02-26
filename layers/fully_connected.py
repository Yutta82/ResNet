import numpy as np


class FullyConnected:
    def __init__(self, in_features, out_features):
        """
        初始化全连接层
        :param in_features: 输入特征的数量
        :param out_features: 输出特征的数量
        """
        self.weights = np.random.randn(in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)

    def forward(self, x):
        """
        前向传播：执行全连接层操作
        :param x: 输入数据，形状为 (N, in_features)
        :return: 输出数据，形状为 (N, out_features)
        """
        self.x = x  # 保存输入
        return np.dot(x, self.weights) + self.bias

    def backward(self, dout):
        """
        反向传播：计算梯度并传递回去
        :param dout: 上一层的梯度
        :return: 传递回去的梯度
        """
        # 计算权重的梯度
        dw = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        # 计算输入的梯度
        dx = np.dot(dout, self.weights.T)

        # 更新权重和偏置
        self.weights -= 0.01 * dw  # 学习率为0.01
        self.bias -= 0.01 * db  # 学习率为0.01

        return dx
