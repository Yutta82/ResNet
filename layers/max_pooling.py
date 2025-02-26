import numpy as np


class MaxPooling:
    def __init__(self, filter_size=2, stride=2, padding=0):
        """
        初始化最大池化层
        :param filter_size: 池化核的大小（默认为 2x2）
        :param stride: 池化的步幅（默认为 2）
        :param padding: 填充（默认为 0）
        """
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        前向传播：执行最大池化操作
        :param x: 输入数据，形状为 (N, C, H, W)
        :return: 池化后的输出
        """
        N, C, H, W = x.shape

        # 计算输出的尺寸
        out_h = (H - self.filter_size + 2 * self.padding) // self.stride + 1
        out_w = (W - self.filter_size + 2 * self.padding) // self.stride + 1

        # 添加填充
        x_padded = np.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)],
                          mode='constant')

        # 输出初始化
        out = np.zeros((N, C, out_h, out_w))
        self.arg_max = np.zeros_like(out, dtype=int)

        # 执行池化操作
        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        # 池化区域
                        region = x_padded[n, c, i * self.stride:i * self.stride + self.filter_size,
                                 j * self.stride:j * self.stride + self.filter_size]
                        out[n, c, i, j] = np.max(region)
                        self.arg_max[n, c, i, j] = np.argmax(region)  # 保存最大值的位置
        return out

    def backward(self, dout):
        """
        反向传播：计算梯度并传递回去
        :param dout: 上一层的梯度
        :return: 传递回去的梯度
        """
        N, C, H, W = dout.shape
        dx = np.zeros_like(dout)

        # 将梯度按照最大值的位置传递
        for n in range(N):
            for c in range(C):
                for i in range(H):
                    for j in range(W):
                        # 反向传播时，将梯度传递给池化窗口中最大值的位置
                        region = dout[n, c, i, j]
                        idx = self.arg_max[n, c, i, j]
                        dx[n, c, i * self.stride:i * self.stride + self.filter_size,
                        j * self.stride:j * self.stride + self.filter_size] = region
        return dx
