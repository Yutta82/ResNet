import numpy as np


class Sigmoid:
    def __init__(self):
        self.out = None  # 保存前向传播的输出，用于反向传播

    def forward(self, input_data):
        """
        前向传播：对输入数据应用 ReLU 激活函数
        :param input_data: 输入数据，形状为 (N, C, H, W)
        :return: 输出数据，形状与输入相同，但应用了 ReLU 激活函数
        """
        self.out = 1 / (1 + np.exp(-input_data))  # Sigmoid 函数
        return self.out

    def backward(self, dout):
        """
        反向传播：计算 ReLU 激活函数的梯度
        :param dout: 来自上游层的梯度，形状与输入相同
        :return: 反向传播的梯度，形状与输入相同
        """
        dx = dout * self.out * (1 - self.out)  # Sigmoid 的梯度
        return dx
