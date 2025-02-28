from layers.batch_norm import BatchNorm
from layers.conv_layer import ConvLayer
from layers.relu_layer import ReLU


class BottleneckBlock:
    def __init__(self, in_channels, out_channels, stride=1):
        """
        初始化瓶颈块（Bottleneck Block）
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 步幅
        """
        # 第一层 1x1 卷积，用于减少通道数
        self.conv1 = ConvLayer(in_channels, out_channels // 4, filter_size=1, stride=1, padding=0)
        self.bn1 = BatchNorm()

        # 第二层 3x3 卷积，用于卷积操作
        self.conv2 = ConvLayer(out_channels // 4, out_channels // 4, filter_size=3, stride=stride, padding=1)
        self.bn2 = BatchNorm()

        # 第三层 1x1 卷积，用于恢复通道数
        self.conv3 = ConvLayer(out_channels // 4, out_channels, filter_size=1, stride=1, padding=0)
        self.bn3 = BatchNorm()

        # 如果输入输出通道数不同，则需要使用 1x1 卷积调整输入通道数
        if in_channels != out_channels:
            self.shortcut = ConvLayer(in_channels, out_channels, filter_size=1, stride=stride, padding=0)
            self.shortcut_bn = BatchNorm()
        else:
            self.shortcut = None

    def forward(self, x):
        """
        前向传播：执行瓶颈块操作
        :param x: 输入数据，形状为 (N, C, H, W)
        :return: 输出数据，形状为 (N, out_channels, H', W')
        """
        identity = x

        # 第1层：1x1卷积，减少通道数
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = ReLU().forward(out)

        # 第2层：3x3卷积，特征提取
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        out = ReLU().forward(out)

        # 第3层：1x1卷积，恢复通道数
        out = self.conv3.forward(out)
        out = self.bn3.forward(out)

        # 残差连接
        if self.shortcut is not None:
            identity = self.shortcut.forward(x)
            identity = self.shortcut_bn.forward(identity)

        out += identity  # 残差连接
        out = ReLU().forward(out)
        return out

    def backward(self, dout):
        """
        反向传播：计算梯度并传递回去
        :param dout: 上一层的梯度
        :return: 传递回去的梯度
        """
        # 反向传播时需要处理梯度
        # 残差连接的梯度传递
        dx = dout

        # 第3层 1x1卷积的梯度
        dx = ReLU().backward(dx)
        dx = self.bn3.backward(dx)
        dx = self.conv3.backward(dx)

        # 第2层 3x3卷积的梯度
        dx = ReLU().backward(dx)
        dx = self.bn2.backward(dx)
        dx = self.conv2.backward(dx)

        # 第1层 1x1卷积的梯度
        dx = ReLU().backward(dx)
        dx = self.bn1.backward(dx)
        dx = self.conv1.backward(dx)

        # 如果有 shortcut，需要计算shortcut部分的梯度
        if self.shortcut is not None:
            identity_grad = self.shortcut.backward(dout)

            # 将梯度与原始输入的梯度相加
            dx += identity_grad

        return dx
