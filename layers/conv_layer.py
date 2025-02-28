import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    将输入数据转换为列形式以便执行卷积操作
    :param input_data: 输入数据，形状为 (N, C, H, W)
    :param filter_h: 滤波器的高度
    :param filter_w: 滤波器的宽度
    :param stride: 步幅
    :param pad: 填充大小
    :return: 转换为列的形状 (N * out_h * out_w, C * filter_h * filter_w)
    """
    N, C, H, W = input_data.shape
    # 计算输出的高度和宽度
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 为输入数据添加填充
    img = np.pad(input_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    # 生成列矩阵，形状为 (N, C, filter_h, filter_w, out_h, out_w)
    # 使用float32是因为float64会大量占用内存
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=np.float32)

    # 将输入数据展开为列的形式
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 将多维数组转换为二维数组 (N * out_h * out_w, C * filter_h * filter_w)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col  # 形状应为 (N * out_h * out_w, C * filter_h * filter_w)


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    将列矩阵转换为图像形式
    :param col: 从 `im2col` 转换得到的列形式矩阵
    :param input_shape: 输入数据的形状，(N, C, H, W)
    :param filter_h: 滤波器的高度
    :param filter_w: 滤波器的宽度
    :param stride: 步幅
    :param pad: 填充大小
    :return: 重建后的图像数据
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 将列矩阵重新变回多维数组
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # 初始化一个零矩阵用于保存结果
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    # 将列形式数据转换回图像
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # 去掉填充部分
    return img[:, :, pad:H + pad, pad:W + pad]


class ConvLayer:
    def __init__(self, in_channels, out_channels, filter_size=3, stride=1, padding=1):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数 原卷积核数量 num_filters
        :param filter_size: 卷积核的大小
        :param stride: 步幅
        :param padding: 填充
        """
        self.input_data = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        # 卷积核的初始化使用 He 初始化方法（适用于 ReLU 激活函数）
        # 高斯分布的均值为 0，方差为 2 / 输入通道数
        self.filters = np.random.randn(out_channels, in_channels, filter_size, filter_size) * np.sqrt(2. / in_channels)
        self.bias = np.zeros(out_channels)  # 偏置初始化为 0

    def forward(self, input_data):
        """
        前向传播计算卷积
        :param input_data: 输入数据，形状为 (N, C, H, W)
        :return: 卷积结果，形状为 (N, num_filters, H_out, W_out)
        """
        self.input_data = input_data
        N, C, H, W = input_data.shape

        # 计算输出高度和宽度
        out_h = (H + 2 * self.padding - self.filter_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.filter_size) // self.stride + 1

        # 使用 im2col 将输入数据转换为列形式
        col = im2col(input_data, self.filter_size, self.filter_size, self.stride, self.padding)

        # 将 filters 转换为列
        col_filter = self.filters.reshape(self.out_channels, -1).T

        print(f"col.shape:{col.shape}")
        print(f"col_filter.shape:{col_filter.shape}")

        # 执行卷积：列数据与卷积核进行矩阵乘法
        out = np.dot(col, col_filter) + self.bias
        out = out.reshape(N, out_h, out_w, self.out_channels)
        out = out.transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        """
        反向传播：计算梯度并传递回去
        :param dout: 上一层的梯度（从下一层传递过来的梯度，形状为 (N, out_channels, H_out, W_out)）
        :return: 传递回去的梯度（dinput）、卷积核的梯度（dfilters）和偏置项的梯度（dbias）
        """

        # 第一步：计算卷积核和偏置的梯度
        # `dout` 传递回来的梯度是上一层的输出梯度，形状为 (N, out_channels, H_out, W_out)
        # 我们需要将 `dout` 展平后与卷积核进行点乘，得到卷积核（filters）的梯度
        # 先把 `dout` 转换为矩阵形式（N * out_h * out_w, out_channels）
        dcol = np.dot(dout.reshape(dout.shape[0], -1), self.filters.reshape(self.out_channels, -1))

        # 第二步：通过 `col2im` 将卷积核的梯度（`dcol`）转换回输入数据的形状
        # 计算输入数据的梯度（dinput），`dcol` 为卷积核梯度，通过 im2col 进行反向传播
        dinput = col2im(dcol, self.input_data.shape, self.filter_size, self.filter_size, self.stride, self.padding)

        # 第三步：计算卷积核（filters）的梯度
        # 卷积核的梯度是通过反向传播传递过来的 `dout` 和对应位置的输入数据的乘积
        # `dcol.T` 将梯度和输入数据列进行点乘，得到卷积核的梯度
        dfilters = dcol.T.reshape(self.out_channels, self.input_data.shape[1], self.filter_size, self.filter_size)

        # 第四步：计算偏置（bias）的梯度
        # 偏置的梯度是 `dout` 在所有输出通道和空间位置上的和，即对每个输出通道求和
        # `np.sum(dout, axis=(0, 2, 3), keepdims=True)`：对 `dout` 的每个输出通道（`axis=0`），
        # 高度（`axis=2`）和宽度（`axis=3`）进行求和，得到每个卷积核的偏置梯度
        dbias = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        # 返回计算得到的梯度（输入数据的梯度、卷积核的梯度、偏置的梯度）
        return dinput, dfilters, dbias

