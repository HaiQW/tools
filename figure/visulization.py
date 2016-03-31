#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
"""visuliza and explore the synthetic data set"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def get_data(data_set, label_set, size_set):
    """从原始数据集合中取特定标签的特定数量的数据样本, 默认数据集合的最后一列
    为数据样本的label"""
    if len(label_set) != len(size_set):
        return None
    count = 0
    dim = data_set.shape[1]
    target = np.zeros(shape=(sum(size_set), dim))
    for idx in range(0, len(label_set)):
        label = label_set[idx]
        size = size_set[idx]
        target[count:count+size, :] \
            = np.reshape(random.sample(data_set[data_set[:, -1] == label, :], size), newshape=(size, dim))
        count += size
    return target


def scatter_2d(data_set):
    """2d散点图,画label propatation算法的可视化效果. data_set 是包含了标签的数据样本集合.
    红色三角形表示rare category,蓝色圆形表示major category."""
    fig = plt.figure()
    labels = np.unique(data_set[:, -1])
    plt.scatter(data_set[data_set[:, -1] == labels[0], 0], data_set[data_set[:, -1] == labels[0], 1], c='red', s=30, marker='^', edgecolor='')
    plt.scatter(data_set[data_set[:, -1] == labels[1], 0], data_set[data_set[:, -1] == labels[1], 1], c='blue', s=15, marker='o', edgecolor='')
    return fig


def main():
    """Main 函数"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_data_1.txt')

    # 选取部分数据画图，并保存
    if not os.path.isfile(file_path):
        origin_data = np.loadtxt('synthetic_data.txt')
        data_set = get_data(origin_data, [2, 5], [30, 500])
        print file_path
        np.savetxt(file_path, data_set)
    else:
        data_set = np.loadtxt(file_path)

    # 散点图1
    fig_scatter_2d = scatter_2d(data_set[:, [0, 1, 3]])
    plt.show()


if __name__ == '__main__':
    main()
