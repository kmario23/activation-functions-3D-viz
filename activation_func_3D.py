"""
Author: @kmario23
A simple script to visualize common activation functions (i.e. non-linearities) used in Neural Networks.
Gives good intuition for how the values of logits would behave after a particular non-linearity is applied.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_activation_func(inp=1.0, activ_func=lambda x: x, name='non-linearity'):
    """
    Function to perform 3D plot of a given activation function
    """
    ws = np.arange(-5, 5, 0.2)
    bs = np.arange(-5, 5, 0.2)

    X, Y = np.meshgrid(ws, bs)
    lst = [activ_func(torch.tensor(w * inp + b)) for w, b in zip(X.flatten(), Y.flatten())]
    os = np.array(lst)
    Z = os.reshape(X.shape)

    # plot figure
    fig = plt.figure()
    fig.suptitle('3D plot of ' + name + ' non-linearity', fontsize=16, backgroundcolor='C5', color='C9')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='green', antialiased=True)
    plt.show()


def main():
    # plot sigmoid
    plot_activation_func(1, torch.sigmoid, 'Sigmoid')

    # plot log-sigmoid
    plot_activation_func(1, F.logsigmoid, 'Log-Sigmoid')

    # plot ReLU
    plot_activation_func(1, torch.relu, 'ReLU')

    # plot RReLU
    plot_activation_func(1, F.rrelu, 'Randomized ReLU')

    # plot Leaky ReLU
    plot_activation_func(1, F.leaky_relu, 'Leaky ReLU')

    # plot ELU
    plot_activation_func(1, F.elu, 'ELU')

    # plot SELU
    plot_activation_func(1, F.selu, 'SELU')

    # plot hardshrink
    plot_activation_func(1, F.hardshrink, 'HardShrink')

    # plot tanhshrink
    plot_activation_func(1, F.tanhshrink, 'TanHShrink')

    # plot softsign
    plot_activation_func(1, F.softsign, 'SoftSign')

    # plot softplus
    plot_activation_func(1, F.softplus, 'SoftPlus')

    # plot softmin
    plot_activation_func(1, F.softmin, 'SoftMin')

    # plot softmax
    plot_activation_func(1, F.softmax, 'SoftMax')

    # plot log-softmax
    plot_activation_func(1, F.log_softmax, 'LogSoftmax')

    # plot gumbel-softmax
    plot_activation_func(1, F.gumbel_softmax, 'Gumbel-SoftMax')

    # plot tanh
    plot_activation_func(1, torch.tanh, 'TanH')

    # plot hardtanh
    plot_activation_func(1, F.hardtanh, 'Hard TanH')


if __name__ == '__main__':
    main()
