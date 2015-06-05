# draw weight
import pickle as pkl
import numpy as np

import theano
import theano.tensor as T
from pylearn2.utils import serial
from pylearn2.datasets import dense_design_matrix
from theano import function
from my_preprocessors import Normal_modif
from my_preprocessors import Adversarial_modif
import draw_images

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import *




class Train_tupple():

    def __init__(self, neurons, eps_train, m_type):
        if m_type == "MNIST":
            self.models_path = "mem/mlp_MNIST_"
        elif m_type == "CIFAR-10":
            self.models_path = "mem/mlp_cifar10_"

        self.neurons = neurons
        self.eps_train = eps_train
        self.model = serial.load(self.model_path())

    # path of tupple
    def model_path(self):
        return self.models_path+str(self.eps_train)+"_"+str(self.neurons)+".pkl"

    # text corresponding to tupple
    def text(self):
        return str(self.neurons)+"_"+str(self.eps_train)


class Plotting_MNIST():

    def __init__(self, save=False):
        nb_neurons = [25,50,100, 200, 400, 800, 1000, 1200, 1400, 2000]
        eps_train_s = [.0,.25]
        self.train_tuples = [Train_tupple(a,b,"MNIST") for a in nb_neurons for b in eps_train_s]
        # self.train_norm = Train_tupple(800,.25,"MNIST_norm")
        self.nb_models = len(self.train_tuples)

        self.x, self.y = self.load_test_set()

    # loads x and y
    def load_test_set(self):
        x_path = '/home/marc/data/mnist_test_X.pkl'
        y_path = '/home/marc/data/mnist_test_y.pkl'
        x = pkl.load(open(x_path))
        y = pkl.load(open(y_path))
        return x,y

    # returns the models with parameters
    def get_models(self):
        # returns stg like (model, neurons, eps_train)
        for train_t in self.train_tuples:
            yield train_t.model, train_t.neurons, train_t.eps_train

    # returns the model with parameters
    def get_model(self, train_t):
        # returns stg like (model, neurons, eps_train)
        if train_t is not None:
            return serial.load(train_t.model_path()), train_t.neurons, train_t.eps_train

    # returns the predictions array (all the p vectors)
    def predictions(self,model, x=None):
        if x is None:
            x = self.x
        
        return model.fprop(theano.shared(x, name='inputs')).eval()

    # gives the overall confidences of the predictions
    def confidence(self, model ,x=None):
        if x is None:
            x = self.x
        
        predictions = self.predictions(model, x)
        confidence = np.max(predictions, axis=1)
        return confidence.mean()*100

    # get missclassification rate of given model
    def accuracy(self, model, x=None):
        if x is None:
            x = self.x
        
        pred = self.predictions(model, x)
        pred = np.argmax(pred,axis=1)
        pred = pred.reshape((-1,1))
        acc  = sum(np.equal(self.y,pred))/float(self.y.size)
        return acc*100

    # returns a confidance / acc array for each of the models
    def array_conf_acc(self):
        # double list with : [(data_id), acc, confidence]
        results = []

        for model, neurons, eps_train in self.get_models():
            results.append([
                (neurons, eps_train),
                self.accuracy(model),
                self.confidence(model) ] )
        return results

    ##########################
    # Test here
    ##########################

    # test a model with adv_datasets and norm_datasets
    def test_array(self, train_t):
        epss = [.0, .05, .1, .15, .2, .25, .3]
        tests = []
        for eps in epss:
            adv_x , y = self.new_dataset(eps, train_t.model, 'adversarial', train_t.eps_train)
            norm_x, y = self.new_dataset(eps, train_t.model, 'normal',      train_t.eps_train)
            tests.append([eps, self.accuracy(train_t.model, adv_x), self.accuracy(train_t.model, norm_x)])
        
        return tests

    # compute a modified dataset
    def new_dataset(self, eps_to_modif, model, m_type, eps_learning=.0):
        ds = dense_design_matrix.DenseDesignMatrix(X=self.x, y=self.y, y_labels=10)

        if m_type == 'adversarial':
            ds.apply_preprocessor(preprocessor=Adversarial_modif(model, eps_learning, eps_to_modif), can_fit=True)
        elif m_type == 'normal':
            ds.apply_preprocessor(preprocessor=Normal_modif('norm', eps_to_modif), can_fit=True)

        return ds.X, self.y

    ##########################
    # Draw here
    ##########################

    # plots the array [ [(data_id), data1, data2] ]
    def histogram_plot(self, data, save_name = None):
        # x and y axis
        plt.figure()
        x = []
        for i in range(self.nb_models):
            if i%2 == 0: # 
                i += .2
            x.append(i)

        x = np.array(x)
        y1 = np.array([tmp[1] for tmp in data])
        y2 = np.array([tmp[2] for tmp in data])
        y_min = min(y1.min(), y2.min()) -2


        # plot the bars
        plt.bar(x, +y1-y_min, facecolor='#9999ff', edgecolor='white')
        plt.bar(x, -y2+y_min, facecolor='#ff9999', edgecolor='white')

        # print the values
        for _x,_y1,_y2 in zip(x,y1,y2):
            plt.text(_x+0.4,  _y1-y_min+0.05, '%.1f' % _y1, ha='center', va= 'bottom')
            plt.text(_x+0.4, -_y2+y_min-0.05, '%.1f' % _y2, ha='center', va= 'top')

        # print the text
        for x,nb_neurons in zip(range(self.nb_models/2), [25,50,100,200, 400, 800, 1000, 1200, 1400, 2000]):
            plt.text(x*2+1, 0, '%s' % nb_neurons, ha='center', va= 'bottom')

        plt.text(-.6,  2, 'Accuracy',   rotation='vertical', verticalalignment='center')
        plt.text(-.6, -3, 'Confidence', rotation='vertical', verticalalignment='center')


        # set axis
        plt.xlim(-1,self.nb_models+.5), xticks([])
        plt.ylim(-np.max(y2-y_min)-5,np.max(y1-y_min)+5), yticks([])

        if save_name is not None:
            plt.savefig('./mem/bar_'+save_name+'.png', dpi=100)
        else:
            plt.show()

    # plots the tests results (on norm and )
    def plot_test(self, train_t_1, train_t_2=None, train_t_3=None, save_name = None):
        # get test results
        tests_result = plotting_instance.test_array(train_t_1)

        plt.figure()
        x  = np.array([tmp[0] for tmp in tests_result])
        y1 = np.array([tmp[1] for tmp in tests_result])
        y2 = np.array([tmp[2] for tmp in tests_result])

        plt.plot(x, y1, x, y2)
        plt.axis([0, .3 , 0, 100])

        if train_t_2 is not None:
            tests_result = plotting_instance.test_array(train_t_2)
            y1 = np.array([tmp[1] for tmp in tests_result])
            y2 = np.array([tmp[2] for tmp in tests_result])

            plt.plot(x, y1, x, y2, linestyle ='dashed')
            plt.axis([0, .3 , 0, 100])

        if train_t_3 is not None:
            tests_result = plotting_instance.test_array(train_t_3)
            y1 = np.array([tmp[1] for tmp in tests_result])
            y2 = np.array([tmp[2] for tmp in tests_result])

            plt.plot(x, y1, x, y2, linestyle ='dash_dot')
            plt.axis([0, .3 , 0, 100])

        if save_name is not None:
            plt.savefig('./mem/plot_test_'+save_name+'.png', dpi=100)
        else:
            plt.show()

    # draw the 25 first inputs of mnist
    def draw_inputs(self, x=None):
        if x is None:
            x = self.x

        draw_images.show_mnist_x(x[0:25], save_path=None)








def get_weights():
    w1 = model.layers[0].get_weights()  # (784, 1200)
    b1 = model.layers[0].get_biases()
    w2 = model.layers[1].get_weights()  # (1200, 10)
    w2 = model.layers[1].get_weights()[:,1]


if __name__ == "__main__":

    plotting_instance = Plotting_MNIST(save=True)

    results = plotting_instance.array_conf_acc()

    # Compare amounts of neurons
    plotting_instance.histogram_plot(results, "acc_all")

    # Compare non-adv-learn VS adv-learn
    plotting_instance.plot_test(plotting_instance.train_tuples[0], plotting_instance.train_tuples[1], save_name="800")

    # Compare non-adv-learn VS adv-learn VS norm-learn
    # plotting_instance.plot_test(plotting_instance.train_tuples[0], plotting_instance.train_tuples[1], plotting_instance.train_norm, save_name="norm_800")
