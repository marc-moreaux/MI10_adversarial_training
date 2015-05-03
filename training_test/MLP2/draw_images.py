import cPickle

import matplotlib.pyplot as plt
from pylab import cm



def show_mnist_x(X, save_path=None):
    if X.shape == (784,):
        plt.figure()
        my_mat = X.reshape([28,28])
        nr, nc = my_mat.shape
        extent = [-0.5, nc-0.5, nr-0.5, -0.5]
        plt.imshow(my_mat, 
            extent=extent, 
            origin='upper', 
            cmap=cm.gray, 
            interpolation='nearest')
        plt.colorbar()
    elif X.shape == (25,784):
        for i in range(5*5):
            plt.subplot(5, 5, i)
            my_mat = X[i].reshape([28,28])
            nr, nc = my_mat.shape
            extent = [-0.5, nc-0.5, nr-0.5, -0.5]
            plt.imshow(my_mat, 
                extent=extent, 
                origin='upper', 
                cmap=cm.gray, 
                interpolation='nearest')

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.cla()


def draw_mnist(save_path = None):
    pkl_file = open("mnist_train_adv.pkl")
    test_set = cPickle.load(pkl_file)
    plt.figure()
    for i in range(5*5):
        plt.subplot(5, 5, i)
        my_mat = test_set.X[i].reshape([28,28])
        nr, nc = my_mat.shape
        extent = [-0.5, nc-0.5, nr-0.5, -0.5]
        plt.imshow(my_mat, extent=extent, origin='upper', cmap=cm.gray, interpolation='nearest')
        plt.colorbar()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.cla()
    pkl_file.close()