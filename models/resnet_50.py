from collections import OrderedDict

import numpy as np

from layers.batch_norm import BatchNorm
from layers.bottleneck_block import BottleneckBlock
from layers.conv_layer import ConvLayer
from layers.fully_connected import FullyConnected
from layers.global_average_pooling import GlobalAveragePooling
from layers.max_pooling import MaxPooling
from layers.relu_layer import ReLU


class ResNet50:
    def __init__(self, num_classes=43):
        self.num_classes = num_classes
        self.build()

    def build(self):
        """
        构建 ResNet-50 网络结构
        """
        self.layers = OrderedDict()

        # 初始卷积层，接受3通道的RGB图像
        self.layers['Conv1'] = ConvLayer(3, 64, filter_size=7, stride=2, padding=3)
        self.layers['BN1'] = BatchNorm()
        self.layers['ReLU1'] = ReLU()

        # 最大池化层
        self.layers['MaxPool'] = MaxPooling(filter_size=3, stride=2, padding=1)

        # 残差块（瓶颈块），显式地定义每个 BottleneckBlock
        # Block1: 64 -> 256 (3 BottleneckBlocks)
        self.layers['Block1_1'] = BottleneckBlock(64, 256, stride=1)
        self.layers['Block1_2'] = BottleneckBlock(256, 256, stride=1)
        self.layers['Block1_3'] = BottleneckBlock(256, 256, stride=1)

        # Block2: 256 -> 512 (4 BottleneckBlocks)
        self.layers['Block2_1'] = BottleneckBlock(256, 512, stride=2)
        self.layers['Block2_2'] = BottleneckBlock(512, 512, stride=1)
        self.layers['Block2_3'] = BottleneckBlock(512, 512, stride=1)
        self.layers['Block2_4'] = BottleneckBlock(512, 512, stride=1)

        # Block3: 512 -> 1024 (6 BottleneckBlocks)
        self.layers['Block3_1'] = BottleneckBlock(512, 1024, stride=2)
        self.layers['Block3_2'] = BottleneckBlock(1024, 1024, stride=1)
        self.layers['Block3_3'] = BottleneckBlock(1024, 1024, stride=1)
        self.layers['Block3_4'] = BottleneckBlock(1024, 1024, stride=1)
        self.layers['Block3_5'] = BottleneckBlock(1024, 1024, stride=1)
        self.layers['Block3_6'] = BottleneckBlock(1024, 1024, stride=1)

        # Block4: 1024 -> 2048 (3 BottleneckBlocks)
        self.layers['Block4_1'] = BottleneckBlock(1024, 2048, stride=2)
        self.layers['Block4_2'] = BottleneckBlock(2048, 2048, stride=1)
        self.layers['Block4_3'] = BottleneckBlock(2048, 2048, stride=1)

        # 全局平均池化
        self.layers['AvgPool'] = GlobalAveragePooling()

        # 全连接层（最终分类层）
        self.layers['FC'] = FullyConnected(2048, self.num_classes)

    def predict(self, x):
        """
        预测：通过前向传播获得预测结果
        :param x: 输入数据，形状为 (N, C, H, W)
        :return: 预测结果
        """
        for layer_name, layer in self.layers.items():
            print(f"Forward:Layer {layer_name}: input shape {x.shape}")
            x = layer.forward(x)
            print(f"Forward:Layer {layer_name}: output shape {x.shape}")
        return x  # 返回最终输出

    def loss(self, y_pred, y_true):
        """
        计算交叉熵损失
        :param y_pred: 模型的预测结果
        :param y_true: 真实标签
        :return: 交叉熵损失
        """
        epsilon = 1e-10  # 避免对数中出现零
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # 防止对数出现负值
        N = y_pred.shape[0]  # 批大小
        return -np.sum(y_true * np.log(y_pred)) / N  # 交叉熵损失公式

    def accuracy(self, y_pred, y_true):
        """
        计算准确率
        :param y_pred: 模型的预测结果
        :param y_true: 真实标签
        :return: 准确率
        """
        predicted_class = np.argmax(y_pred, axis=1)
        true_class = np.argmax(y_true, axis=1)
        return np.mean(predicted_class == true_class)

    def gradient(self, x, y_true):
        """
        计算梯度：反向传播每一层的梯度
        :param x: 输入数据
        :param y_true: 真实标签
        :return: 每一层的梯度
        """
        # 计算前向传播得到的预测值
        y_pred = self.predict(x)

        # 计算损失对输出的梯度
        dout = y_pred - y_true

        # 反向传播
        for layer_name, layer in reversed(self.layers.items()):
            print(f"Backward:Layer {layer_name}: input shape {dout.shape}")
            dout = layer.backward(dout)
            print(f"Backward:Layer {layer_name}: output shape {dout.shape}")
        return dout
