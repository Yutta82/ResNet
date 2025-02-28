import numpy as np

"""
强制性地调整激活值分布广度
具体就是进行使数据分布的均值为0方差为1的正规化
"""


class BatchNorm:
    def __init__(self, epsilon=1e-5, momentum=0.9):
        """
        初始化批归一化层
        :param epsilon: 避免除零错误的极小值
        :param momentum: 用于更新均值和方差的动量
        """
        self.input_data = None
        self.x_hat = None
        self.var = None
        self.mean = None
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = 1.0  # 标准化缩放系数（会在forward中初始化）
        self.beta = 0.0  # 标准化偏移量（会在forward中初始化）
        self.running_mean = None  # 存储均值，用于推理时的使用
        self.running_var = None  # 存储方差，用于推理时的使用

    def forward(self, input_data, training=True):
        """
        前向传播过程，计算归一化后的输出
        :param input_data: 输入数据，形状为(N, C, H, W)
        :param training: 是否为训练模式，影响均值和方差的计算
        :return: 归一化后的输出
        """
        # 获取输入数据的形状
        N, C, H, W = input_data.shape

        # 计算均值和方差
        if training:
            # 计算当前批次的均值和方差
            self.mean = np.mean(input_data, axis=(0, 2, 3), keepdims=True)
            self.var = np.var(input_data, axis=(0, 2, 3), keepdims=True)

            # 更新滑动平均值（用于推理）
            if self.running_mean is None:
                self.running_mean = self.mean
                self.running_var = self.var
            else:
                self.running_mean = (self.momentum * self.mean
                                     + (1 - self.momentum) * self.running_mean)
                self.running_var = (self.momentum * self.var
                                    + (1 - self.momentum) * self.running_var)
        else:
            # 在推理时使用训练时计算的均值和方差
            self.mean = self.running_mean
            self.var = self.running_var

        # 标准化
        self.x_hat = (input_data - self.mean) / np.sqrt(self.var + self.epsilon)

        # 如果没有初始化gamma和beta，则在训练时进行初始化
        if self.gamma is None:
            self.gamma = np.ones((1, C, 1, 1))  # 对每个通道应用相同的缩放因子
            self.beta = np.zeros((1, C, 1, 1))  # 对每个通道应用相同的偏移量

        # 返回标准化后的数据
        out = self.gamma * self.x_hat + self.beta
        return out

    def backward(self, dout):
        """
        反向传播计算梯度
        :param dout: 上一层传递来的梯度
        :return: 当前层的梯度以及gamma和beta的梯度
        """
        # 获取输入数据的形状
        N, C, H, W = dout.shape

        # 计算 beta 和 gamma 的梯度
        dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
        dgamma = np.sum(dout * self.x_hat, axis=(0, 2, 3), keepdims=True)

        # 计算 x_hat 的梯度
        dx_hat = dout * self.gamma

        # 计算输入数据的梯度
        dvar = np.sum(dx_hat * (self.input_data - self.mean) * -0.5 * np.power(self.var + self.epsilon, -1.5),
                      axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(self.var + self.epsilon), axis=(0, 2, 3), keepdims=True) + dvar * np.sum(
            -2 * (self.input_data - self.mean), axis=(0, 2, 3), keepdims=True) / N

        # 计算输入的梯度
        dx = dx_hat / np.sqrt(self.var + self.epsilon) + dvar * 2 * (self.input_data - self.mean) / N + dmean / N
        return dx, dgamma, dbeta
