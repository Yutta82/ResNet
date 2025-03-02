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
        self.conv1 = ConvLayer(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0)
        self.bn1 = BatchNorm()
        self.relu1 = ReLU()

        # 第二层 3x3 卷积，用于卷积操作
        self.conv2 = ConvLayer(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1)
        self.bn2 = BatchNorm()
        self.relu2 = ReLU()

        # 第三层 1x1 卷积，用于恢复通道数
        self.conv3 = ConvLayer(out_channels // 4, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = BatchNorm()
        self.relu3 = ReLU()

        # 如果输入输出通道数不同，则需要使用 1x1 卷积调整输入通道数
        if in_channels != out_channels:
            self.shortcut = ConvLayer(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
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
        out = self.relu1.forward(out)

        # 第2层：3x3卷积，特征提取
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        out = self.relu2.forward(out)

        # 第3层：1x1卷积，恢复通道数
        out = self.conv3.forward(out)
        out = self.bn3.forward(out)

        # 残差连接
        if self.shortcut is not None:
            identity = self.shortcut.forward(x)
            identity = self.shortcut_bn.forward(identity)

        out += identity  # 残差连接
        out = self.relu3.forward(out)
        return out

    def backward(self, dout):
        """
        反向传播：计算梯度并传递回去
        :param dout: 上一层的梯度
        :return: 传递回去的梯度
        """
        # 首先通过最后的 ReLU 反向传播
        dout = self.relu3.backward(dout)

        # 残差连接的梯度分为两条路径
        dx_main = dout  # 主路径的梯度
        dx_identity = dout.copy()  # 残差路径的梯度

        # 主路径的反向传播
        # 第三层
        dx = dx_main
        dx, dweights, dbias = self.bn3.backward(dx)
        dx, dweights, dbias = self.conv3.backward(dx)
        # 第二层
        dx = self.relu2.backward(dx)
        dx, dweights, dbias = self.bn2.backward(dx)
        dx, dweights, dbias = self.conv2.backward(dx)
        # 第一层
        dx = self.relu1.backward(dx)
        dx, dweights, dbias = self.bn1.backward(dx)
        dx, dweights, dbias = self.conv1.backward(dx)

        # 残差路径的反向传播
        if self.shortcut is not None:
            dx_identity, dweights_identity, dbias_identity = self.shortcut_bn.backward(dx_identity)
            dx_identity, dweights_identity, dbias_identity = self.shortcut.backward(dx_identity)
        # 如果没有 shortcut，则 dx_identity 不变（即 dout）

        # 将两条路径的梯度相加
        dx += dx_identity
        return dx
