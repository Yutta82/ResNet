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
        动量（Momentum） 和 速度（Velocity）
        是为了加速梯度下降过程并改善优化算法的收敛性，
        特别是在面对局部最小值或者高斯噪声时，
        动量可以帮助避免震荡并加速收敛。
        """
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])

            # 应用权重衰减
            """
            权重衰减（Weight Decay） 是一种正则化方法，用于防止模型过拟合。
            权重衰减通过将模型的权重加入到梯度中来实现。
            在这里，通过将当前模型参数（params[key]）乘以一个衰减系数 (self.weight_decay)，然后加到梯度中，
            确保在更新时惩罚较大的权重，鼓励模型学习较小的权重。
                (当一个权重很大，通过式一可以把对应的梯度也变大，
                然后在计算速度时需要减去梯度，梯度越大减得越多，
                得到的对应的速度就越小，最后加到参数上的值就越小)
            这样可以有效避免模型在训练时过度依赖某些特征。
            """
            grads[key] += self.weight_decay * params[key]  # 式一

            # 计算动量更新
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            params[key] += self.velocity[key]

        return params
