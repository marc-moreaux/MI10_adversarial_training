

from pylearn2.utils import serial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


for nhid in [2,5,25,100,200,400,800,1000,1200,1400,1800,2000]:
    for eps_adv in [.0,.25]:

        model = serial.load("./mem/SA_mlp_MNIST_"+str(nhid)+"_"+str(eps_adv)+".pkl")

        plt.figure()

        nb_plots = 10
        for i in range(nb_plots):
            plt.subplot(1, nb_plots, i+1)
            w1_1 = model.layers[0].layer_content.get_weights() # size : (in, nb_neurons)
            w2_1 = model.layers[1].get_weights()[:,i] # size : (nb_neurons, 1)
            w_shape = int(math.sqrt(w1_1.shape[0]))
            tmp = w1_1.dot(w2_1)
            plt.imshow(tmp.reshape((w_shape,w_shape)), cmap = cm.Greys_r)
            plt.axis('off')



        plt.savefig("./mem/img_weights_"+str(nhid)+"_"+str(eps_adv)+".png", dpi=100)