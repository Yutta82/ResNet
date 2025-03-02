import numpy as np

from layers.relu_layer import ReLU
from layers.sigmod_layer import Sigmoid


class SEBlock:
    def __init__(self, channel, reduction=16, learning_rate=0.01):
        """
        初始化 SE-Block
        :param channel: 输入通道数
        :param reduction: 压缩比例，通常设为 16 或 8
        """
        self.learning_rate = None
        self.channel = channel
        self.reduction = reduction

        # 初始化全连接层权重和偏置
        self.fc1_weights = np.random.randn(channel, channel // reduction) * 0.1  # 第一层全连接权重
        self.fc1_bias = np.zeros(channel // reduction)  # 第一层偏置
        self.fc2_weights = np.random.randn(channel // reduction, channel) * 0.1  # 第二层全连接权重
        self.fc2_bias = np.zeros(channel)  # 第二层偏置

        # 实例化 ReLU 和 Sigmoid
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

        self.input = None
        self.pooled = None  # 存储全局平均池化结果
        self.fc1_out = None  # 存储 fc1 的输出
        self.fc2_out = None  # 存储 fc2 的输出
        self.scale = None  # 存储加权比例

    def forward(self, x):
        """
        前向传播：应用 SE-Block
        :param x: 输入数据，形状为 (N, C, H, W)
        :return: 注意力加权后的输入
        """
        self.input = x
        N, C, H, W = x.shape

        # 全局平均池化
        self.pooled = np.mean(x, axis=(2, 3), keepdims=True)

        # 第一层全连接
        fc1_out = np.dot(self.pooled.reshape(N, C), self.fc1_weights)
        fc1_out = np.maximum(0, fc1_out)  # ReLU 激活
        self.fc1_out = fc1_out

        # 第二层全连接
        fc2_out = np.dot(fc1_out, self.fc2_weights)
        # scale = 1 / (1 + np.exp(-fc2_out))  # Sigmoid 激活
        scale = self.sigmoid.forward(fc2_out)
        self.fc2_out = fc2_out
        self.scale = scale.reshape(N, C, 1, 1)  # 存储scale，用于反向传播

        # 扩展维度，进行加权
        scale = scale.reshape(N, C, 1, 1)
        return x * scale  # 加权输入

    def backward(self, dout):
        """
        反向传播：计算梯度并传递回去
        :param dout: 上一层的梯度
        :return: 传递回去的梯度
        """
        N, C, H, W = self.input.shape

        # 计算加权比例的梯度
        dscale = dout * self.scale  # 加权的梯度
        dscale = dscale.sum(axis=(0, 2, 3), keepdims=True)  # 对H和W维度求和

        # 计算输入的梯度 dx
        dx = dout * self.scale

        # 计算 scale 的梯度
        dscale = (dout * self.input).sum(axis=(2, 3), keepdims=True)

        # Sigmoid 的反向传播
        dsigma = self.sigmoid.backward(dscale.reshape(N, C))  # 调用 Sigmoid 的反向传播

        # fc2 的梯度
        dfc2_out = dsigma
        dfc2_weights = np.dot(self.fc1_out.T, dfc2_out)
        dfc2_bias = np.sum(dfc2_out, axis=0)

        # ReLU 的反向传播
        dfc1_out = np.dot(dfc2_out, self.fc2_weights.T)
        dfc1_out = self.relu.backward(dfc1_out)  # 调用 ReLU 的反向传播

        # fc1 的梯度
        dfc1_weights = np.dot(self.pooled.reshape(N, C).T, dfc1_out)
        dfc1_bias = np.sum(dfc1_out, axis=0)

        # 更新 fc2 权重和偏置
        self.fc2_weights -= self.learning_rate * dfc2_weights
        self.fc2_bias -= self.learning_rate * dfc2_bias

        # 更新 fc1 权重和偏置
        self.fc1_weights -= self.learning_rate * dfc1_weights
        self.fc1_bias -= self.learning_rate * dfc1_bias

        # 全局平均池化的梯度
        dpooled = np.dot(dfc1_out, self.fc1_weights.T).reshape(N, C, 1, 1)
        dx += dpooled / (H * W)

        return dx
