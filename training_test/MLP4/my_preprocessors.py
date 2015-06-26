import os
import sys
import pickle
import costAdv
import pickle as pkl
import theano.tensor as T

from theano import function
from theano.tensor.shared_randomstreams import RandomStreams

from pylearn2.utils import serial, sharedX
from pylearn2.space import CompositeSpace, NullSpace, VectorSpace
from pylearn2.datasets import cifar10, preprocessing, dense_design_matrix
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.format.target_format import convert_to_one_hot
from pylearn2.datasets.preprocessing import Preprocessor

import scipy.ndimage.interpolation


class Adversarial_modif(Preprocessor):
    """
    Modify the inputs such that they are modified following the worst case.

    Parameters
    ----------
    window_shape : WRITEME
    """

    def __init__(self, model, learning_eps, training_eps=0.07):
        self.__dict__.update(locals())
        del self.self
        self.model = model
        self.learning_eps = learning_eps
        self.training_eps = training_eps

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            bla bla
        """
        # Get X and Y from the dataset
        X = dataset.X
        y = convert_to_one_hot(dataset.y).squeeze()

        # Compute the symbolic function
        eps = self.training_eps
        X_var = T.TensorType(broadcastable=[s == 1 for s in X.shape],dtype=X.dtype)()
        y_var = T.TensorType(broadcastable=[s == 1 for s in y.shape],dtype=y.dtype)()
        eps_var = T.dscalar('eps')
        data = (X_var, y_var)

        # X = X + eps(sign(grad_x(J))) 
        # Bounded between 0 and 1
        X_adv = eps_var * T.sgn( T.grad(self.model.cost_from_X(data), X_var))
        f = function([X_var,y_var,eps_var], T.clip(X_var +  X_adv, 0, 1) )

        # Apply the function
        dataset.X = f( X, y, eps )


class Normal_modif(Preprocessor):
    def __init__(self, distr_type='norm', training_eps=0.07):
        self.__dict__.update(locals())
        del self.self
        self.distr_type = distr_type
        self.training_eps = training_eps

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            bla bla
        """
        # Get X from the dataset
        X = dataset.X

        # Declare all the variables
        X_var = T.TensorType(broadcastable=[s == 1 for s in X.shape],dtype=X.dtype)()
        eps = self.training_eps
        eps_var = T.dscalar('eps')
        srng = RandomStreams(seed=234)

        distr = None
        if self.distr_type == 'norm':
            distr = srng.normal(X.shape)
        elif self.distr_type == 'uni':
            distr = srng.uniform(X.shape)

        # X = X + eps( noise )
        # Bound X between 0 and 1
        noise = eps_var * distr
        f = function([X_var, eps_var], T.clip(X_var +  noise, 0, 1) )

        # Apply the function
        dataset.X = f( X, eps )



# Not working !
class Other_modif(Preprocessor):
    def __init__(self, distr_type='rot', training_eps=0.07):
        self.__dict__.update(locals())
        del self.self
        self.distr_type = distr_type
        self.training_eps = training_eps

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            bla bla
        """
        # Get X from the dataset
        X = dataset.X

        # Declare all the variables
        X_var = T.TensorType(broadcastable=[s == 1 for s in X.shape],dtype=X.dtype)()
        eps = self.training_eps
        eps_var = T.dscalar('eps')

        
        if self.distr_type == 'rot':
            f = function([X_var, eps_var], T.clip( scipy.ndimage.interpolation.rotate(X_var.reshape(28,28),eps_var) ).reshape(28*28,-1) )

        # Apply the function
        dataset.X = f( X, eps )





def make_adv_training_set(learning_eps, training_eps=0.07):

    print "=====> Making adversarial dataset for {"+str(learning_eps)+"; and with "+str(training_eps)+"}"

    train_data_path = "/home/marc/data/"
    X = pickle.load( open( os.path.join(train_data_path, "mnist_test_X.pkl" )))
    y = pickle.load( open( os.path.join(train_data_path, "mnist_test_y.pkl" )))
    train = dense_design_matrix.DenseDesignMatrix(X=X, y=y, y_labels=10)
    model = serial.load("mlp"+str(learning_eps)+".pkl")

    train.apply_preprocessor(preprocessor=Adversarial_input_modif(model,learning_eps, training_eps), can_fit=True)
    train.use_design_loc('mnist_train_adv.npy')
    train_pkl_path = 'mnist_train_adv.pkl'
    serial.save(train_pkl_path, train)



if __name__ == "__main__":

    # Open dataset
    test_X = pkl.load(open("/home/marc/data/mnist_test_X.pkl"))
    test_y = pkl.load(open("/home/marc/data/mnist_test_y.pkl"))
    ds = dense_design_matrix.DenseDesignMatrix(X=test_X, y=test_y, y_labels=10)
    ds.apply_preprocessor(
        preprocessor=Normal_modif('uni', .3),
        can_fit=True)
    
