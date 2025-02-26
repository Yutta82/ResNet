import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []

    def update(self, train_loss, train_accuracy, val_loss=None, val_accuracy=None):
        """
        收集每个epoch的训练损失、训练准确率、验证损失和验证准确率
        :param train_loss: 当前训练集的损失
        :param train_accuracy: 当前训练集的准确率
        :param val_loss: 当前验证集的损失
        :param val_accuracy: 当前验证集的准确率
        """
        # 添加训练集数据
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_accuracy)

        # 添加验证集数据（如果有的话）
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if val_accuracy is not None:
            self.val_accuracy.append(val_accuracy)

    def show(self):
        """
        在训练结束后，绘制最终的损失和准确率图
        """
        # 创建图形窗口
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 设置损失值图表
        ax1.set_title('Loss over Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.plot(self.train_loss, label='Train Loss', color='blue')
        if self.val_loss:
            ax1.plot(self.val_loss, label='Validation Loss', color='red')
        ax1.legend()

        # 设置准确率图表
        ax2.set_title('Accuracy over Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.plot(self.train_accuracy, label='Train Accuracy', color='blue')
        if self.val_accuracy:
            ax2.plot(self.val_accuracy, label='Validation Accuracy', color='red')
        ax2.legend()

        plt.show()
