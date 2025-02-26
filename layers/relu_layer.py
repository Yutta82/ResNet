import numpy as np


class ReLU:
    def __init__(self):
        self.input_data = None

    def forward(self, input_data):
        """
        前向传播：对输入数据应用 ReLU 激活函数
        :param input_data: 输入数据，形状为 (N, C, H, W)
        :return: 输出数据，形状与输入相同，但应用了 ReLU 激活函数
        """
        # 存储输入数据用于反向传播
        self.input_data = input_data

        # ReLU 激活：大于 0 的值保留，小于 0 的值置为 0
        return np.maximum(0, input_data)

    def backward(self, dout):
        """
        反向传播：计算 ReLU 激活函数的梯度
        :param dout: 来自上游层的梯度，形状与输入相同
        :return: 反向传播的梯度，形状与输入相同
        """
        # ReLU 的梯度：对于大于 0 的输入，梯度为 1；对于小于等于 0 的输入，梯度为 0
        return dout * (self.input_data > 0)
