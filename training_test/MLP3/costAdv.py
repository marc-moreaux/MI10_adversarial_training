import theano.tensor as T
import pickle
import os
from functools import wraps
from pylearn2.utils import serial, sharedX
from pylearn2.space import CompositeSpace, NullSpace, VectorSpace
from pylearn2.datasets import cifar10, preprocessing, dense_design_matrix
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.utils.data_specs import DataSpecsMapping



class AdversarialCost(DefaultDataSpecsMixin, Cost):
    """The default Cost to use with an MLP.

    It simply calls the MLP's cost_from_X method.
    """

    supervised = True

    def __init__(self, learning_eps):
        self.learning_eps = learning_eps
        

    def expr(self, model, data, **kwargs):
        """Returns a theano expression for the cost function.

        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """

        space, sources = self.get_data_specs(model)
        space.validate(data)
        X, y = data
        alpha = .5
        adv_X = X + self.learning_eps * T.sgn(T.grad(model.cost_from_X(data), X))
        adv_data = (adv_X, y)

        return alpha*model.cost_from_X(data) + (1-alpha)*model.cost_from_X(adv_data)

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return False



