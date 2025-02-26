import numpy as np


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        初始化 Adam 优化器
        :param learning_rate: 学习率
        :param beta1: 一阶矩估计的衰减率
        :param beta2: 二阶矩估计的衰减率
        :param epsilon: 防止除零的极小值
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0  # 时间步长

    def update(self, params, grads):
        """
        更新模型参数
        :param params: 模型参数（字典）
        :param grads: 梯度（字典）
        :return: 更新后的参数
        """
        self.t += 1

        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            # 更新一阶矩和二阶矩
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # 计算偏差修正
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # 更新参数
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params
