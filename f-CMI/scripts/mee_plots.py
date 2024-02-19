import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from nnlib.nnlib.matplotlib_utils import set_default_configs

set_default_configs(plt, seaborn)


class NestedDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


def plot_ss(ax, xs, data, mode):
    ax.plot(xs, data[:, 0, 0], label='Error', marker='o')
    ax.fill_between(xs, data[:, 0, 0]-data[:, 0, 1], data[:, 0, 0]+data[:, 0, 1], alpha=0.2)
    if mode == 0:
        ax.plot(xs, data[:, 1, 0], label='Square-Root', marker='x')
        ax.fill_between(xs, data[:, 1, 0]-data[:, 1, 1], data[:, 1, 0]+data[:, 1, 1], alpha=0.2)
        ax.plot(xs, data[:, 2, 0], label='Fast-Rate', marker='^')
        ax.fill_between(xs, data[:, 2, 0]-data[:, 2, 1], data[:, 2, 0]+data[:, 2, 1], alpha=0.2)
        ax.plot(xs, data[:, 5, 0], label='Binary KL', marker='d')
    elif mode == 1:
        ax.plot(xs, data[:, 3, 0], label='Weighted', marker='x')
        ax.fill_between(xs, data[:, 3, 0]-data[:, 3, 1], data[:, 3, 0]+data[:, 3, 1], alpha=0.2)
        ax.plot(xs, data[:, 2, 0], label='Fast-Rate', marker='^')
        ax.fill_between(xs, data[:, 2, 0]-data[:, 2, 1], data[:, 2, 0]+data[:, 2, 1], alpha=0.2)
        ax.plot(xs, data[:, 5, 0], label='Binary KL', marker='d')
    elif mode == 2:
        ax.plot(xs, data[:, 3, 0], label='Adaptive', marker='x')
        ax.fill_between(xs, data[:, 3, 0]-data[:, 3, 1], data[:, 3, 0]+data[:, 3, 1], alpha=0.2)
        ax.plot(xs, data[:, 4, 0], label='Universal', marker='^')
        ax.fill_between(xs, data[:, 4, 0]-data[:, 4, 1], data[:, 4, 0]+data[:, 4, 1], alpha=0.2)


def plot_mnist_ss_n(mode):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    results_dir = "results"
    exp_name = "fcmi-mnist-4vs9-CNN"
    results_file_path = os.path.join(results_dir, exp_name, 'mee_results.pkl')
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    ns = [75, 250, 1000, 4000]
    data = np.zeros((4, 6, 2))
    for i in range(4):
        results_n = results[ns[i]][200]
        bounds = results_n['mee_bound']
        gen_gap = results_n['exp_val_acc'] - results_n['exp_train_acc']

        data[i, 0, :] = np.mean(gen_gap), np.std(gen_gap)
        for j in range(4):
            data[i, j + 1, :] = np.mean(bounds[j]), np.std(bounds[j])
        data[i, 5, :] = bounds[4], 0

    xs = np.arange(4)
    plot_ss(ax, xs, data, mode)

    ax.set_xlabel('n')
    ax.set_ylabel('Error')
    ax.set_xticks(xs)
    ax.set_xticklabels(ns)
    ax.legend()

    fig.savefig('figures/mnist_ss_n_%d.pdf' % mode, format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_cifar_ss_n(mode):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    results_dir = "results"
    exp_name = "cifar10-pretrained-resnet50"
    results_file_path = os.path.join(results_dir, exp_name, 'mee_results.pkl')
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    ns = [1000, 5000, 20000]
    data = np.zeros((3, 6, 2))
    for i in range(3):
        results_n = results[ns[i]][40]
        bounds = results_n['mee_bound']
        gen_gap = results_n['exp_val_acc'] - results_n['exp_train_acc']

        data[i, 0, :] = np.mean(gen_gap), np.std(gen_gap)
        for j in range(4):
            data[i, j + 1, :] = np.mean(bounds[j]), np.std(bounds[j])
        data[i, 5, :] = bounds[4], 0

    xs = np.arange(3)
    plot_ss(ax, xs, data, mode)

    ax.set_xlabel('n')
    ax.set_ylabel('Error')
    ax.set_xticks(xs)
    ax.set_xticklabels(ns)
    ax.legend()

    fig.savefig('figures/cifar_ss_n_%d.pdf' % mode, format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_mnist_ss_epoch(mode):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    results_dir = "results"
    exp_name = "fcmi-mnist-4vs9-CNN-LD"
    results_file_path = os.path.join(results_dir, exp_name, 'mee_results.pkl')
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    ns = np.arange(1, 11) * 4
    data = np.zeros((10, 6, 2))
    for i in range(10):
        results_n = results[4000][ns[i]]
        bounds = results_n['mee_bound']
        gen_gap = results_n['exp_val_acc'] - results_n['exp_train_acc']

        data[i, 0, :] = np.mean(gen_gap), np.std(gen_gap)
        for j in range(4):
            data[i, j + 1, :] = np.mean(bounds[j]), np.std(bounds[j])
        data[i, 5, :] = bounds[4], 0

    xs = np.arange(10)
    plot_ss(ax, xs, data, mode)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.set_xticks(xs)
    ax.set_xticklabels(ns)
    ax.legend()

    fig.savefig('figures/mnist_ss_epoch_%d.pdf' % mode, format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_mnist_loo_n():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    results_dir = "results"
    exp_name = "fcmi-mnist-4vs9-CNN-loo"
    results_file_path = os.path.join(results_dir, exp_name, 'mee_results.pkl')
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    ns = [75, 250, 1000, 4000]
    data = np.zeros((4, 2, 2))
    for i in range(4):
        results_n = results[ns[i]][200]
        bounds = np.sort(results_n['mee_bound'])
        gen_gap = np.sort(results_n['exp_val_acc'] - results_n['exp_train_acc'])

        data[i, 0, :] = np.nanmean(gen_gap), np.nanstd(gen_gap)
        data[i, 1, :] = np.nanmean(bounds), np.nanstd(bounds)

    xs = np.arange(4)
    ax.plot(xs, data[:, 0, 0], label='Error', marker='o')
    # ax.fill_between(xs, data[:, 0, 0]-data[:, 0, 1], data[:, 0, 0]+data[:, 0, 1], alpha=0.2)
    ax.plot(xs, data[:, 1, 0], label='Square-Root', marker='x')
    # ax.fill_between(xs, data[:, 1, 0]-data[:, 1, 1], data[:, 1, 0]+data[:, 1, 1], alpha=0.2)

    ax.set_xlabel('n')
    ax.set_ylabel('Error')
    ax.set_xticks(xs)
    ax.set_xticklabels(ns)
    ax.legend()

    fig.savefig('figures/mnist_loo_n.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_cifar_loo_n():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    results_dir = "results"
    exp_name = "cifar10-pretrained-resnet50-loo"
    results_file_path = os.path.join(results_dir, exp_name, 'mee_results.pkl')
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    ns = [1000, 5000, 20000]
    data = np.zeros((3, 2, 2))
    for i in range(3):
        results_n = results[ns[i]][40]
        bounds = results_n['mee_bound']
        gen_gap = results_n['exp_val_acc'] - results_n['exp_train_acc']

        data[i, 0, :] = np.mean(gen_gap), np.std(gen_gap)
        data[i, 1, :] = np.nanmean(bounds), np.nanstd(bounds)

    xs = np.arange(3)
    ax.plot(xs, data[:, 0, 0], label='Error', marker='o')
    # ax.fill_between(xs, data[:, 0, 0]-data[:, 0, 1], data[:, 0, 0]+data[:, 0, 1], alpha=0.2)
    ax.plot(xs, data[:, 1, 0], label='Square-Root', marker='x')
    # ax.fill_between(xs, data[:, 1, 0]-data[:, 1, 1], data[:, 1, 0]+data[:, 1, 1], alpha=0.2)

    ax.set_xlabel('n')
    ax.set_ylabel('Error')
    ax.set_xticks(xs)
    ax.set_xticklabels(ns)
    ax.legend()

    fig.savefig('figures/cifar_loo_n.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_mnist_loo_epoch():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    results_dir = "results"
    exp_name = "fcmi-mnist-4vs9-CNN-LD-loo"
    results_file_path = os.path.join(results_dir, exp_name, 'mee_results.pkl')
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    ns = np.arange(1, 11) * 4
    data = np.zeros((10, 2, 2))
    for i in range(10):
        results_n = results[4000][ns[i]]
        bounds = results_n['mee_bound']
        gen_gap = results_n['exp_val_acc'] - results_n['exp_train_acc']

        data[i, 0, :] = np.mean(gen_gap), np.std(gen_gap)
        data[i, 1, :] = np.mean(bounds), np.std(bounds)

    xs = np.arange(10)
    ax.plot(xs, data[:, 0, 0], label='Error', marker='o')
    ax.fill_between(xs, data[:, 0, 0]-data[:, 0, 1], data[:, 0, 0]+data[:, 0, 1], alpha=0.2)
    ax.plot(xs, data[:, 1, 0], label='Square-Root', marker='x')
    ax.fill_between(xs, data[:, 1, 0]-data[:, 1, 1], data[:, 1, 0]+data[:, 1, 1], alpha=0.2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.set_xticks(xs)
    ax.set_xticklabels(ns)
    ax.legend()

    fig.savefig('figures/mnist_loo_epoch.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_c():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    L = np.arange(1, 100) / 100
    C = -np.log(2 - np.exp(np.log(2) * L)) / (np.log(2) * L) - 1
    ax.plot(L, C)

    ax.set_xlabel('$L_i$')
    ax.set_ylabel(r'$C_i \times L_i$')

    fig.savefig('figures/adaptive_c.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


if __name__ == '__main__':
    # for i in range(3):
    #     plot_mnist_ss_n(i)
    #     plot_cifar_ss_n(i)
    #     plot_mnist_ss_epoch(i)

    # plot_mnist_loo_n()
    # plot_cifar_loo_n()
    # plot_mnist_loo_epoch()

    plot_c()