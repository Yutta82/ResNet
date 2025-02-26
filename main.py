from models.resnet_50 import ResNet50

from optimizers.adam import Adam
from optimizers.sgd import SGD
from utils import plotter
from utils.data_preprocessing import create_train_test_data

# 加载 GTSRB 数据集
data_dir = "path/to/GTSRB/training"  # GTSRB 数据集的训练数据路径
X_train, X_test, y_train, y_test = create_train_test_data(data_dir)

# 初始化 ResNet-50 模型
model = ResNet50(num_classes=43)

# 初始化优化器
# optimizer = Adam(learning_rate=0.001)  # 或者使用 SGD
optimizer = SGD(learning_rate=0.01, momentum=0.9)

# 训练模型
epochs = 10
for epoch in range(epochs):
    # 前向传播计算预测
    y_pred = model.predict(X_train)

    # 计算损失
    loss = model.loss(y_pred, y_train)

    # 计算准确率
    accuracy = model.accuracy(y_pred, y_train)

    # 计算梯度
    gradients = model.gradient(X_train, y_train)

    # 使用优化器更新模型的参数
    model.layers = optimizer.update(model.layers, gradients)

    # 更新 Plotter
    plotter.update(loss, accuracy)

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

# 评估模型
y_pred_test = model.predict(X_test)
test_loss = model.loss(y_pred_test, y_test)
test_accuracy = model.accuracy(y_pred_test, y_test)

print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
# 最后展示图表
plotter.show()

