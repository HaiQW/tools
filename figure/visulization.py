#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
"""visuliza and explore the synthetic data set"""
import os
import random
import numpy as np
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
        target[count:count+size, :] = np.reshape(random.sample(data_set[data_set[:, -1] == label, :], size), newshape=(size, dim))
        count += size
    return target


def scatter_2d(data_set):
    """2d散点图,画label propatation算法的可视化效果. data_set 是包含了标签的数据样本集合.
    红色三角形表示rare category,蓝色圆形表示major category."""
    labels = np.unique(data_set[:, -1])
    major_set = data_set[data_set[:, -1] == labels[1], :]
    rare_set = data_set[data_set[:, -1] == labels[0], :]

    # 稀有类和主要类的标签列表
    idx_1 = [2, 13, 15, 1]
    idx_2 = [0, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19]
    idx_3 = []
    idx_4 = []
    for idx in range(0, major_set.shape[0]):
        if major_set[idx][0] < -5 and major_set[idx][1] > 2:
            idx_3.append(idx)
        else:
            idx_4.append(idx)

    # 标签传播算法做图: 2幅子图
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7.5))
    axes[0].scatter(data_set[:, 0], data_set[:, 1], s=5, c='grey', edgecolor='grey', label=r'unlabeled example')
    axes[0].scatter(rare_set[4, 0], rare_set[4, 1], c='red', s=100, marker='^', edgecolor='red', label=r'seed example')
    axes[1].scatter(rare_set[4, 0], rare_set[4, 1], c='red', s=100, marker='^', edgecolor='red')
    axes[1].scatter(rare_set[idx_2, 0], rare_set[idx_2, 1], s=5, c='grey', edgecolor='grey', label=r'unlabeled example')
    axes[1].scatter(rare_set[idx_1, 0], rare_set[idx_1, 1], c='', s=50, marker='^', edgecolor='red', label=r'labeled rare example')
    axes[1].scatter(major_set[idx_3, 0], major_set[idx_3, 1], s=50, c='', edgecolor='blue', label=r'labeled major example')
    axes[1].scatter(major_set[idx_4, 0], major_set[idx_4, 1], s=5, c='grey', edgecolor='grey')
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[1].set_yticklabels([])
    axes[0].set_xlim([-8, 8])
    axes[0].set_ylim([-12, 9])
    axes[1].set_xlim([-8, 8])
    axes[1].set_ylim([-12, 9])
    axes[1].legend(loc="lower right")
    axes[0].legend(loc="lower right")
    return fig


def fig_fscore():
    """fscore柱状图"""
    algorithms = ("mknn-lp", "RNN", "rco-svm", "one-class svm", "RACH")
    x = np.arange(len(algorithms))
    width = 0.40
    fscores = [0.9, 0.7, 0.5, 0.4, 0.1]
    fig = plt.bar(x ,fscores, width, edgecolor="blue", linewidth=2, color="",  align='center')
    plt.xticks(x+width/2.0, algorithms)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('f-score')
    return fig


def main():
    """Main 函数"""
    file_path = os.path.join( \
            os.path.dirname(os.path.abspath(__file__)), 'synthetic_data_1.txt')

    # 选取部分数据画图，并保存
    if not os.path.isfile(file_path):
        origin_data = np.loadtxt('synthetic_data.txt')
        data_set = get_data(origin_data, [2, 5], [20, 250])
        print file_path
        np.savetxt(file_path, data_set)
    else:
        data_set = np.loadtxt(file_path)

    # 散点图1
    if not os.path.isfile('label_propagation_fig1.eps'):
        fig_scatter_2d = scatter_2d(data_set[:, [0, 1, 3]])
        plt.savefig(os.path.dirname(file_path) + '/label_propagation_fig1.eps', \
            format='eps', bbox_inches='tight', dpi=1200)
        plt.show()

    # 柱状图
    if not os.path.isfile('label_propagation_fig2.eps'):
        fig2 = fig_fscore()
        plt.savefig(os.path.dirname(file_path) + '/label_propagation_fig2.eps', \
            format='eps', bbox_inches='tight', dpi=1200)
        plt.show()


if __name__ == '__main__':
    main()
