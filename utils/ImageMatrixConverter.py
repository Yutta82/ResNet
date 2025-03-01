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
