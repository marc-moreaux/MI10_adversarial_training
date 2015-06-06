import os
import sys
import pickle
import costAdv
import draw_images
import theano.tensor as T

from theano import function
from pylearn2.utils import serial, sharedX
from pylearn2.space import CompositeSpace, NullSpace, VectorSpace
from pylearn2.config import yaml_parse
from pylearn2.datasets import cifar10, preprocessing, dense_design_matrix
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.datasets.preprocessing import Preprocessor

def big_print(message):
    print "\n********************************************************"
    print "==> " + message
    print "********************************************************\n"


def adv_learning(learning_eps, training_eps):

    # Do some prints
    # sys.stdout = open("./out"+str(learning_eps)+"_"+str(training_eps)+".txt", "w")
    msg = "Compute adversarial dataset => " + str(training_eps)
    big_print(msg)
    
    # Compute adversarial dataset
    make_adv_training_set(learning_eps, training_eps)
    draw_images.draw_mnist("mnist_"+str(learning_eps)+"_"+str(training_eps)+".png")

    # Test the model on adversarial dataset
    mlp_test_yaml = open('mlp_test.yaml', 'r').read()
    mlp_hyper_params = {'training_eps': training_eps,
                        'learning_eps': learning_eps,
                        'save_path': '.'}
    mlp_test_yaml = mlp_test_yaml % mlp_hyper_params
    train_obj = yaml_parse.load(mlp_test_yaml)
    train_obj.main_loop()


class Adversarial_input_modification(Preprocessor):
    """
    Modify the inputs such that they are modified following the worst case.

    Parameters
    ----------
    window_shape : WRITEME
    """

    def __init__(self, weight_path, learning_eps, training_eps=0.07):
        self.__dict__.update(locals())
        del self.self
        self.weight_path = weight_path
        self.learning_eps = learning_eps
        self.training_eps = training_eps

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            Loads the model and its cost. Compute "\frac{\partial J}{\partial x}".
        """
        # Instanciate model and cost
        model = pickle.load(open(self.weight_path))
        cost = costAdv.AdversarialCost(self.learning_eps)

        # *****
        # Get X and y (y in matrix mode)
        # *****
        X = dataset.X
        y = None
        data_specs = cost.get_data_specs(model)
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)

        train_iteration_mode = 'shuffled_sequential'
        rng = None
        flat_data_specs = (CompositeSpace(space_tuple), source_tuple)

        iterator = dataset.iterator(mode=train_iteration_mode,
                                    batch_size=10000,
                                    num_batches=1,
                                    data_specs=flat_data_specs,
                                    return_tuple=True,
                                    rng=rng)
        
        for data in iterator:
            y = data[source_tuple.index('targets')]


        # Compute the symbolic function
        eps = self.training_eps
        X_var = T.TensorType(broadcastable=[s == 1 for s in X.shape],dtype=X.dtype)()
        y_var = T.TensorType(broadcastable=[s == 1 for s in y.shape],dtype=y.dtype)()
        eps_var = T.dscalar('eps')
        data = (X_var, y_var)

        # X = X + eps(sign(grad_x(J)))
        f = function([X_var,y_var,eps_var], X_var + eps_var * T.sgn( T.grad(model.cost_from_X(data), X_var) ))

        # Apply the function
        dataset.X = f( X, y, eps )


def make_adv_training_set(learning_eps, training_eps=0.07):

    print "=====> Making adversarial dataset for {"+str(learning_eps)+"; and with "+str(training_eps)+"}"

    train_data_path = "/home/marc/data/"
    X = pickle.load( open( os.path.join(train_data_path, "mnist_test_X.pkl" )))
    y = pickle.load( open( os.path.join(train_data_path, "mnist_test_y.pkl" )))
    train = dense_design_matrix.DenseDesignMatrix(X=X, y=y, y_labels=10)

    train.apply_preprocessor(preprocessor=Adversarial_input_modification("mlp"+str(learning_eps)+".pkl",learning_eps, training_eps), can_fit=True)
    train.use_design_loc('mnist_train_adv.npy')
    train_pkl_path = 'mnist_train_adv.pkl'
    serial.save(train_pkl_path, train)


#
# Kind of main thingy
#
for learning_eps in [.0]:#, .25 ,.3]: #[.0, .1, .2, .25,  .3] :

    sys.stdout = open("./out"+str(learning_eps)+".txt", "a")

    # big_print("train ADVERSARIAL ")
    # mlp_yaml = open('mlp.yaml', 'r').read()
    # mlp_hyper_params = {'learning_eps': learning_eps, 'save_path' : '.'}
    # mlp_yaml = mlp_yaml % mlp_hyper_params
    # train_obj = yaml_parse.load(mlp_yaml)
    # train_obj.main_loop()

    adv_learning(learning_eps, .007)
    adv_learning(learning_eps, .1)
    adv_learning(learning_eps, .2)
    adv_learning(learning_eps, .25)
    adv_learning(learning_eps, .3)


