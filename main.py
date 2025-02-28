import os

import numpy as np

from models.resnet_50 import ResNet50
from optimizers.sgd import SGD
from utils import plotter
from utils.data_preprocessing import create_train_test_data

# 加载 GTSRB 数据集
project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = r"/dataset/train"  # GTSRB 数据集的训练数据路径
X_train, X_test, y_train, y_test = create_train_test_data(project_root + data_dir)
print("Original X_train shape:", X_train.shape)
# 调整数据的维度顺序，从 (N, H, W, C) -> (N, C, H, W)
X_train = X_train.transpose(0, 3, 1, 2)
X_test = X_test.transpose(0, 3, 1, 2)
print("Adjusted X_train shape:", X_train.shape)

# 初始化 ResNet-50 模型
model = ResNet50(num_classes=43)
# 初始化优化器
# optimizer = Adam(learning_rate=0.001)  # 或者使用 SGD
optimizer = SGD(learning_rate=0.01, momentum=0.9)
# 设置批量大小，批量梯度下降中每次处理的数据量
batch_size = 16
# 训练模型
epochs = 1

for epoch in range(epochs):
    # 逐批次训练
    for batch in range(0, len(X_train), batch_size):
        # 取出当前批次的数据和标签
        X_batch = X_train[batch:batch + batch_size]  # 当前批次的输入数据
        y_batch = y_train[batch:batch + batch_size]  # 当前批次的标签
        # 进行前向传播，得到预测结果
        y_pred = model.predict(X_batch)
        # 计算当前批次的损失值，使用交叉熵损失函数
        loss = model.loss(y_pred, y_batch)
        # 计算当前批次的准确率
        accuracy = model.accuracy(y_pred, y_batch)
        # 计算当前批次的梯度（反向传播）
        gradients = model.gradient(X_batch, y_batch)
        # 使用优化器更新模型的参数
        model.layers = optimizer.update(model.layers, gradients)
        # 更新 Plotter
        plotter.update(loss, accuracy)
        # 每个批次的训练过程结束后输出损失和准确率
        print(f"Batch {batch // batch_size + 1}: Loss {loss:.4f}, Accuracy {accuracy:.4f}")
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

# 评估模型
y_pred_test = model.predict(X_test)
test_loss = model.loss(y_pred_test, y_test)
test_accuracy = model.accuracy(y_pred_test, y_test)

print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
# 最后展示图表
plotter.show()
