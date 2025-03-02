import numpy as np

from utils.image_matrix_converter import *


class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数 原卷积核数量 num_kernel
        :param kernel_size: 卷积核的大小
        :param stride: 步幅
        :param padding: 填充
        """
        self.input_data = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.col = None  # 保存im2col的结果

        # 卷积核的初始化使用 He 初始化方法（适用于 ReLU 激活函数）
        # 高斯分布的均值为 0，方差为 2 / 输入通道数
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / in_channels)
        self.bias = np.zeros(out_channels)  # 偏置初始化为 0

    def forward(self, x):
        """
        前向传播计算卷积
        :param x: 输入数据，形状为 (N, C, H, W)
        :return: 卷积结果，形状为 (N, num_filters, H_out, W_out)
        """
        self.input_data = x
        N, C, H, W = x.shape

        # 计算输出高度和宽度
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 使用 im2col 将输入数据转换为列形式
        self.col = im2col(x, self.kernel_size, self.kernel_size, self.stride, self.padding)

        # 将 weight 转换为列
        col_weights = self.weights.reshape(self.out_channels, -1).T

        # 执行卷积：列数据与卷积核进行矩阵乘法
        out = np.dot(self.col, col_weights) + self.bias
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        """
        反向传播：计算梯度并传递回去
        :param dout: 上一层的梯度（从下一层传递过来的梯度，形状为 (N, out_channels, H_out, W_out)）
        :return: 传递回去的梯度（dinput）、卷积核的梯度（dfilters）和偏置项的梯度（dbias）
        """
        # 将 dout 重塑为 (N * out_h * out_w, out_channels)
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # 计算权重的梯度 dweights
        dweights = (np.dot(dout_reshaped.T, self.col)
                    .reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))

        # 计算偏置的梯度 dbias
        dbias = np.sum(dout, axis=(0, 2, 3))

        # 计算输入的梯度 dx
        col_weights = self.weights.reshape(self.out_channels, -1)
        dcol = np.dot(dout_reshaped, col_weights)

        dx = col2im(dcol, self.input_data.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)

        return dx, dweights, dbias
