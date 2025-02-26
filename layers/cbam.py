import numpy as np


class CBAM:
    def __init__(self, channel, reduction=16):
        """
        初始化 CBAM
        :param channel: 输入通道数
        :param reduction: 压缩比例，通常设为 16 或 8
        """
        self.channel = channel
        self.reduction = reduction

        # Channel Attention
        self.channel_fc1 = None  # 第一层全连接
        self.channel_fc2 = None  # 第二层全连接

        # Spatial Attention
        self.spatial_conv = None  # 3x3 卷积层

    def forward(self, x):
        """
        前向传播：应用 CBAM
        :param x: 输入数据，形状为 (N, C, H, W)
        :return: 注意力加权后的输入
        """
        N, C, H, W = x.shape

        # Channel Attention
        # 计算通道注意力
        channel_avg_pool = np.mean(x, axis=(2, 3), keepdims=True)
        channel_max_pool = np.max(x, axis=(2, 3), keepdims=True)

        # 将平均池化和最大池化拼接
        channel_feature = np.concatenate([channel_avg_pool, channel_max_pool], axis=1)

        # 第一层全连接
        channel_fc1_out = np.dot(channel_feature.reshape(N, C * 2), self.channel_fc1)
        channel_fc1_out = np.maximum(0, channel_fc1_out)  # ReLU 激活

        # 第二层全连接
        channel_fc2_out = np.dot(channel_fc1_out, self.channel_fc2)
        channel_scale = 1 / (1 + np.exp(-channel_fc2_out))  # Sigmoid 激活

        # 扩展维度，进行加权
        channel_scale = channel_scale.reshape(N, C, 1, 1)

        # Apply Channel Attention
        x = x * channel_scale  # 加权输入

        # Spatial Attention
        # 计算空间注意力
        spatial_avg_pool = np.mean(x, axis=1, keepdims=True)
        spatial_max_pool = np.max(x, axis=1, keepdims=True)

        spatial_feature = np.concatenate([spatial_avg_pool, spatial_max_pool], axis=1)

        # 3x3 卷积层
        spatial_attention = self.spatial_conv.forward(spatial_feature)
        spatial_attention = 1 / (1 + np.exp(-spatial_attention))  # Sigmoid 激活

        # Apply Spatial Attention
        return x * spatial_attention  # 加权输入
