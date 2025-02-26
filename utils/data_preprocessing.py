import os

import cv2
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_data(data_dir, img_size=(32, 32)):
    """
    加载并预处理数据
    :param data_dir: 数据集路径
    :param img_size: 调整后的图像大小
    :return: 返回处理后的图像和标签
    """
    images = []
    labels = []

    # 遍历每个类别的文件夹
    for label in os.listdir(data_dir):
        class_path = os.path.join(data_dir, label)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)  # 调整图像大小
                images.append(img)
                labels.append(int(label))  # 标签为文件夹的名称
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def preprocess_data(images, labels):
    """
    数据预处理，包括归一化和标签编码
    :param images: 输入图像数据
    :param labels: 输入标签
    :return: 归一化后的图像和独热编码标签
    """
    images = images / 255.0  # 图像归一化到[0, 1]
    labels = to_categorical(labels, num_classes=43)  # 标签转换为独热编码
    return images, labels


def create_train_test_data(data_dir, test_size=0.2):
    """
    创建训练集和测试集
    :param data_dir: 数据集路径
    :param test_size: 测试集占比
    :return: 返回训练集和测试集数据
    """
    images, labels = load_data(data_dir)
    images, labels = preprocess_data(images, labels)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def data_augmentation(X_train):
    """
    对训练数据进行数据增强
    :param X_train: 训练数据
    :return: 增强后的训练数据
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)
    return datagen
