#!/usr/bin/python
#-*-coding:utf-8-*-
import os

from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np


def main():
    """ 用多元高斯产生随机数"""
    if not os.path.isfile('synthetic_gauss.txt'):
        major_dataset = multivariate_normal(mean=[1, 5], cov=[[1, 1], [1, 1.1]], size=200)
        major_labels = np.zeros(shape=[200,1], dtype=np.int)
        rare_dataset = multivariate_normal(mean=[-1, 2.8], cov=[[0.01, 0], [0, 0.01]], size=15)
        rare_labels = np.ones(shape=[15,1], dtype=np.int)
        # 最终人工合成数据集
        final_dataset = np.append(np.append(major_dataset, major_labels, axis=1),
                np.append(rare_dataset, rare_labels, axis=1), axis=0)
        np.savetxt(fname='synthetic_gauss.txt', X=final_dataset)
    else:
        final_dataset = np.loadtxt('synthetic_gauss.txt')

    plt.scatter(final_dataset[0:200, 0], final_dataset[0:200, 1])
    plt.scatter(final_dataset[200:215, 0], final_dataset[200:215, 1], marker='*', c='red', s=80)
    plt.show()


if __name__ == '__main__':
    main()
