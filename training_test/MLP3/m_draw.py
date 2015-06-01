# draw weight
import numpy as np
from pylearn2.utils import serial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import theano.tensor as T
from theano import function
import pickle as pkl

model_1 = serial.load("mem/mlp_COV_0.0.pkl")
model_2 = serial.load("mem/mlp_COV_0.25.pkl")

print min(model_1.monitor.channels['test_y_misclass'].val_record)
print min(model_2.monitor.channels['test_y_misclass'].val_record)


# => 150 -> yes
0.511660728731
0.510628044268

# => 120 -> no
0.510774341233
0.511152992203

# => 100 -> no
0.51174678577
0.511781208585

# => 100 -> yes
0.510800158345
0.510301027521

# => 80 -> yes
0.510757129826
0.510671072787

# => 60 -> no
0.512469664894
0.512521299117

# => 60 -> no
0.510352661744
0.511677940139



def get_weights():
    w1 = model.layers[0].get_weights()  # (784, 1200)
    b1 = model.layers[0].get_biases()
    w2 = model.layers[1].get_weights()  # (1200, 10)
    w2 = model.layers[1].get_weights()[:,1]



# img = np.multiply(w1+b1.T, w2.T).sum(axis=1)
# img = img.reshape((-1,1))

# img = img.reshape(28,28)
# plt.imshow( img, cmap=cm.Greys_r)
# plt.show()




def sample_prop():
    ds = pkl.load(open("/home/marc/data/mnist_test_X.pkl"))

    x = ds[0].reshape((-1,1))
    y = np.array([[1]])

    x_var = T.dmatrix(name='x')
    y_var = T.scalar(name='y')

    print function( [x_var], a.fprop(x_var) )(x.T)

