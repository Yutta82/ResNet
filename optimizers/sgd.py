import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0, weight_decay=0.0):
        """
        初始化 SGD 优化器
        :param learning_rate: 学习率
        :param momentum: 动量
        :param weight_decay: 权重衰减（L2正则化）
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}

    def update(self, params, grads):
        """
        更新模型参数
        :param params: 模型参数（字典）
        :param grads: 梯度（字典）
        :return: 更新后的参数
        """
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])

            # 应用权重衰减
            grads[key] += self.weight_decay * params[key]

            # 计算动量更新
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            params[key] += self.velocity[key]

        return params
