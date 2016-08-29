#!/usr/bin/python
#-*-coding:utf-8-*-
import os

from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.patches as patches

mpl.rcParams["axes.linewidth"] = 3

def filter_dataset():
    """选取部分数据集，用于放大画图"""
    orig_dataset = np.loadtxt('synthetic_gauss.txt', dtype=np.float)
    final_dataset = []
    for i in range(0, 200):
        if orig_dataset[i, 0] <= -0.2 and orig_dataset[i, 1] <= 4:
            final_dataset.extend(orig_dataset[i,:])
    final_dataset = np.reshape(np.array(final_dataset), newshape=(len(final_dataset)/3, 3))
    final_dataset = np.append(final_dataset, orig_dataset[200:215, :], axis=0)
    np.savetxt(fname='filter_dataset.txt', X=final_dataset)


def main():
    """ 用多元高斯产生随机数"""
    file_path = os.path.join( \
            os.path.dirname(os.path.abspath(__file__)), 'synthetic_gauss.txt')
    if not os.path.isfile(file_path):
        major_dataset = multivariate_normal(mean=[1, 5], cov=[[1, 1], [1, 1.1]], size=200)
        major_labels = np.zeros(shape=[200,1], dtype=np.int)
        rare_dataset = multivariate_normal(mean=[-1, 2.8], cov=[[0.01, 0], [0, 0.01]], size=15)
        rare_labels = np.ones(shape=[15,1], dtype=np.int)
        # 最终人工合成数据集
        final_dataset = np.append(np.append(major_dataset, major_labels, axis=1),
                np.append(rare_dataset, rare_labels, axis=1), axis=0)
        np.savetxt(fname='synthetic_gauss.txt', X=final_dataset)
    else:
        final_dataset = np.loadtxt('synthetic_gauss.txt', dtype=np.float)


    # 获取过滤后的数据集合(主要包含稀有类数据点)
    if not os.path.isfile(os.path.dirname(file_path) + '/filter_dataset.txt'):
        filter_dataset()
    filter_dataset = np.loadtxt('filter_dataset.txt')

    # 原始数据分布图
    an = np.linspace(0, 2 * np.pi, 100)
    figure, ax = plt.subplots(1)
    ax.scatter(final_dataset[0:200, 0], final_dataset[0:200, 1], edgecolor='blue', facecolor='', label='major example')
    ax.scatter(final_dataset[200:215, 0], final_dataset[200:215, 1], edgecolor='red', facecolor='', marker='*', s=100, label='rare example')
    ax.plot(0.13 * np.cos(an)-1.03, 0.13*np.sin(an)+2.89, linewidth=1,  c='black')
    ax.plot(0.16 * np.cos(an)-0.84, 0.16*np.sin(an)+3.17, linewidth=1, c='black')
    ax.arrow(-0.98, 2.93, -0.5, 0.5, head_width=0.05, linewidth=1, head_length=0.1, fc='k', ec='k')
    ax.arrow(-0.943, 3.13, -0.13, 0.26, head_width=0.05, linewidth=1, head_length=0.1, fc='k', ec='k')
    ax.text(-1.59, 3.57, 'A', color='k', size=20)
    ax.text(-1.17, 3.51, 'B', color='k', size=20)
    legend = plt.legend(loc="lower right")
    legend.get_frame().set_linewidth(2)
    plt.xticks([])
    plt.yticks([])

    rect = patches.Rectangle((-1.8, 2), 1.5, 1.5, linewidth=2, edgecolor='black', facecolor='none')
    # ax.plot(np.array([-1.8, -4.75]), np.array([3.5, 3.5]));
    ax.add_patch(rect)

    innerfig = plt.axes([0.15, 0.55, .33, .33], axisbg='white')
    innerfig.scatter(filter_dataset[filter_dataset[:, 2] == 0, 0], filter_dataset[filter_dataset[:, 2] == 0, 1], \
                     edgecolor='blue', facecolor='')
    innerfig.scatter(filter_dataset[filter_dataset[:, 2] == 1, 0], filter_dataset[filter_dataset[:, 2] == 1, 1], \
                    edgecolor='red', facecolor='', marker='*', s=100)
    innerfig.plot(0.13 * np.cos(an) - 1.03, 0.13 * np.sin(an) + 2.89, c='black')
    innerfig.plot(0.16 * np.cos(an) - 0.84, 0.16 * np.sin(an) + 3.17, c='black')
    innerfig.arrow(-0.98, 2.93, -0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
    innerfig.arrow(-0.943, 3.13, -0.13, 0.26, head_width=0.05, head_length=0.1, fc='k', ec='k')
    innerfig.text(-1.59, 3.57, 'A', color='k', size=20)
    innerfig.text(-1.17, 3.51, 'B', color='k', size=20)
    plt.xlim(np.min(filter_dataset[filter_dataset[:, 2] == 0, 0]) + 0.2, np.max(filter_dataset[filter_dataset[:, 2] == 0, 0]) - 0.2)
    plt.ylim(np.min(filter_dataset[filter_dataset[:, 2] == 0, 1]) + 0.4, np.max(filter_dataset[filter_dataset[:, 2] == 0, 1]) - 0.2)
    for axis in ['top','bottom','left','right']:
        innerfig.spines[axis].set_linewidth(2)

    #ax.arrow(-1.1, 3.8, 0.1, 1, head_width=0.05, head_length=0.1, fc='k', ec='k')
    ax.annotate("", xy=(-1.1, 5.3), xytext=(0, -60), size=25, textcoords='offset points',
               arrowprops=dict(arrowstyle="simple", fc="0.6", ec="none"))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.dirname(file_path) + '/mknn_lp_mutual_nearest.eps', \
            format='eps', bbox_inches='tight', dpi=1200)
    plt.show()
    plt.close()

    # 过滤部分数据后分布图
    plt.scatter(filter_dataset[filter_dataset[:, 2] == 0, 0], filter_dataset[filter_dataset[:, 2] == 0, 1], \
            edgecolor='blue', facecolor='', label='major example')
    plt.scatter(filter_dataset[filter_dataset[:, 2] == 1, 0], filter_dataset[filter_dataset[:, 2] == 1, 1], \
            edgecolor='red', facecolor='', marker='*', s=100, label='major example')
    plt.plot(0.13 * np.cos(an) - 1.03, 0.13 * np.sin(an) + 2.89, c='black')
    plt.plot(0.16 * np.cos(an) - 0.84, 0.16 * np.sin(an) + 3.17, c='black')
    plt.arrow(-0.98, 2.93, -0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
    plt.arrow(-0.943, 3.13, -0.13, 0.3, head_width=0.05, head_length=0.1, fc='k', ec='k')
    plt.text(-1.59, 3.57, 'a', color='k', size=20)
    plt.text(-1.17, 3.51, 'b', color='k', size=20)
    plt.legend(loc="lower right", fontsize=14)
    plt.savefig(os.path.dirname(file_path) + '/mknn_lp_mutual_nearest_filter.eps', \
            format='eps', bbox_inches='tight', dpi=1200)


if __name__ == '__main__':
    main()
