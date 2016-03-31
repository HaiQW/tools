#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
"""visuliza and explore the synthetic data set"""
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


def scatter_2d(x, y):
    """2d散点图,画label propatation算法的可视化效果. x数据样本,y是数据样本的标签(label)."""
    fig = plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolor='')
    plt.show(fig)
    return


def main():
    """Main 函数"""
    df = pd.read_csv('synthetic_data.txt', delimiter=' ')
    print df.describe

    dataset = get_data(df.values, [2, 5], [50, 1000])
    print dataset.shape
    # fig = plt.figure()
    scatter_2d(dataset[:, 0:2], dataset[:, 3])


if __name__ == '__main__':
    main()
