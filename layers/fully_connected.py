import numpy as np


class FullyConnected:
    def __init__(self, in_features, out_features):
        """
        初始化全连接层
        :param in_features: 输入特征的数量
        :param out_features: 输出特征的数量
        """
        self.original_shape = None
        self.x = None
        self.weights = np.random.randn(in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)

    def forward(self, x):
        """
        前向传播：执行全连接层操作
        :param x: 输入数据，形状为 (N, in_features)
        :return: 输出数据，形状为 (N, out_features)
        """
        self.original_shape = x.shape  # 保存输入的原始形状
        if len(x.shape) > 2:  # 如果输入是 (N, C, 1, 1)
            x = x.reshape(x.shape[0], -1)  # 展平为 (N, C)
        self.x = x  # 保存展平后的输入
        return np.dot(x, self.weights) + self.bias

    def backward(self, dout):
        """
        反向传播：计算梯度并传递回去
        :param dout: 上一层的梯度，形状为 (N, out_features)
        :return: 传递回去的梯度，形状与输入 x 一致
        """
        # 计算权重的梯度
        dw = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        # 计算输入的梯度
        dx = np.dot(dout, self.weights.T)

        # 如果输入的原始形状是 (N, C, 1, 1)，恢复 dx 的形状
        if len(self.original_shape) > 2:
            dx = dx.reshape(self.original_shape)

        # 更新权重和偏置
        self.weights -= 0.01 * dw  # 学习率为0.01
        self.bias -= 0.01 * db  # 学习率为0.01

        return dx
