import numpy as np

from utils.ImageMatrixConverter import *


class MaxPooling:
    def __init__(self, filter_size=2, stride=2, padding=0):
        """
        初始化最大池化层
        :param filter_size: 池化核的大小（默认为 2x2）
        :param stride: 池化的步幅（默认为 2）
        :param padding: 填充（默认为 0）
        """
        self.arg_max = None
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_shape = None  # 保存输入形状

    def forward(self, x):
        """
        前向传播：执行最大池化操作
        :param x: 输入数据，形状为 (N, C, H, W)
        :return: 池化后的输出
        """
        self.input_shape = x.shape
        N, C, H, W = self.input_shape

        # 计算输出的尺寸
        out_h = (H - self.filter_size + 2 * self.padding) // self.stride + 1
        out_w = (W - self.filter_size + 2 * self.padding) // self.stride + 1

        # 将输入转换为列形式 (N*out_h*out_w, C*filter_size*filter_size)
        col = im2col(x, self.filter_size, self.filter_size, self.stride, self.padding)
        # 重塑为 (N * out_h * out_w, C, filter_size * filter_size)
        col = col.reshape(-1, self.filter_size * self.filter_size, C).transpose(0, 2, 1)

        # 找到每个池化区域的最大值及其索引
        self.arg_max = np.argmax(col, axis=2)
        out = np.max(col, axis=2)

        # 重塑为 (N, C, out_h, out_w)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout):
        """
        反向传播：计算梯度并传递回去
        :param dout: 上一层的梯度
        :return: 传递回去的梯度
        """
        N, C, out_h, out_w = dout.shape
        H, W = self.input_shape[2], self.input_shape[3]

        # (N*out_h*out_w, C)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)

        # 初始化梯度矩阵
        col_dx = np.zeros((dout.shape[0], C, self.filter_size * self.filter_size), dtype=np.float32)

        # 矢量化梯度分配
        rows = np.arange(dout.size // C).repeat(C)
        channels = np.tile(np.arange(C), dout.size // C)
        col_dx[rows, channels, self.col_max_idx.ravel()] = dout.ravel()

        # 调整维度顺序以匹配 col2im 的输入格式，变为 (N * out_h * out_w, filter_size * filter_size * C)
        col_dx = col_dx.transpose(0, 2, 1).reshape(-1, C * self.filter_size * self.filter_size)
        dx = col2im(col_dx, self.input_shape, self.filter_size, self.filter_size, self.stride, self.padding)

        # 修复填充处理
        if self.padding > 0:
            dx = dx[:, :, self.padding:H + self.padding, self.padding:W + self.padding]

        return dx
