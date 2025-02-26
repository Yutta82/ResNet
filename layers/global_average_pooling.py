import numpy as np


class GlobalAveragePooling:
    def __init__(self):
        """
        初始化全局平均池化层
        """
        pass

    def forward(self, x):
        """
        前向传播：执行全局平均池化操作
        :param x: 输入数据，形状为 (N, C, H, W)
        :return: 池化后的输出，形状为 (N, C, 1, 1)
        """
        return np.mean(x, axis=(2, 3), keepdims=True)

    def backward(self, dout):
        """
        反向传播：计算梯度并传递回去
        :param dout: 上一层的梯度
        :return: 传递回去的梯度
        """
        # 全局平均池化时，每个位置的梯度都被平均传递给所有空间位置
        return dout / np.prod(dout.shape[2:])
