import math
import theano 
import os.path
import numpy as np
import pickle as pkl
import my_preprocessors as preproc
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pylearn2
from pylearn2.utils import serial
from pylearn2.datasets import dense_design_matrix
from pylearn2.format.target_format import convert_to_one_hot

from my_preprocessors import Normal_modif
from my_preprocessors import Adversarial_modif

from random import randint


class Model_used():
    def __init__(self, nhid=800, learning_eps='.25', db_name="MNIST"):
        self.learning_eps = learning_eps;
        self.nhid = nhid
        self.db_name = db_name

        if db_name == 'CIFAR':
            self.nhid = 2500

        # set name
        self.model_name = str(self.nhid)+"_"+str(self.learning_eps)

        
        # load model
        self.model = self.load()


    def load(self):
        path_model = "./mem/SA_mlp_"+self.db_name+"_"+self.model_name+".pkl"
        if os.path.isfile(path_model):
            model = serial.load(path_model)
        else:
            print "doesn't exist :"
            print path_model

        return model

    # returns the predictions array ()
    def predict(self, x):
        #
        return self.model.fprop(theano.shared(x, name='inputs')).eval()

    # return confidence towards input x
    def confidence(self, x):
        predictions = self.predict(x)
        confidence = np.max(predictions, axis=1)
        return confidence.mean()*100

    # get misclassification rate of given model
    def accuracy(self, data):
        x,y = data
        pred = self.predict(x)
        pred = np.argmax(pred,axis=1)
        pred = pred.reshape((-1,1))
        acc  = sum(np.equal(y,pred))/float(y.size)
        return acc*100

    # return the weights at a given layer
    def get_weights(self, layer):
        if isinstance(self.model.layers[layer], pylearn2.models.mlp.PretrainedLayer):
            return self.model.layers[layer].get_param_values()[2]
        return self.model.layers[layer].get_weights()

    # return the biases at a given layer
    def get_biases(self,layer):
        if isinstance(self.model.layers[layer], pylearn2.models.mlp.PretrainedLayer):
            return self.model.layers[layer].get_param_values()[1]
        return self.model.layers[layer].get_biases()




class Set():
    def __init__(self, which='test', db_name="MNIST"):
        # load test set
        if db_name == 'MNIST':
            x_path = '/home/marc/data/mnist_'+which+'_X.pkl'
            y_path = '/home/marc/data/mnist_'+which+'_y.pkl'
        if db_name == 'MNIST_bin':
            x_path = '/home/marc/data/mnist_'+which+'_bin_X.pkl'
            y_path = '/home/marc/data/mnist_'+which+'_bin_y.pkl'
        if db_name == 'CIFAR':
            x_path = '/media/marc/SAMSUNG_SD_/data/'+which+'_X.pkl'
            y_path = '/media/marc/SAMSUNG_SD_/data/'+which+'_y.pkl'

        self.x = pkl.load(open(x_path))
        self.y = pkl.load(open(y_path))


    def get_data(self):
        #
        return (self.x,self.y)

    def modified_dataset(self, m_type=None ,eps_to_modif=.25, model_used=None):
        ds = dense_design_matrix.DenseDesignMatrix(X=self.x, y=self.y, y_labels=10)

        if m_type == 'adv':
            ds.apply_preprocessor(preprocessor=Adversarial_modif(model_used.model, model_used.learning_eps, eps_to_modif), can_fit=True)
        elif m_type == 'norm':
            ds.apply_preprocessor(preprocessor=Normal_modif('norm', eps_to_modif), can_fit=True)
        elif m_type == 'none':
            return ds.X, self.y

        return ds.X, self.y

    def extract_class(self, class_nb):
        if type(class_nb) is int:
            mask = np.equal(self.y, class_nb)
        return self.x[mask], self.y[mask]



def data_extract_classes(classes_array=[1,2]):


    return data


def plot_neurons_impact(save_name=None):
    # Prepare data
    tmp = Set('test')
    data = tmp.get_data()

    a = Model_used(100, .0)
    a.accuracy(data)

    models_used = []
    acc = []
    conf = []
    for nhid in [2,5,25,50,100,400,800,1200,2000]:
        for eps_adv in [.0, .25]:
            tmp = Model_used(nhid, eps_adv)
            models_used.append(tmp)
            acc.append(tmp.accuracy(data))
            conf.append(tmp.confidence(data[0]))

    nb_models = len(models_used)
    x = []
    for i in range(nb_models):
        if i%2 == 0: # 
            i += .2
        x.append(i)

    # plot the figure
    plt.figure()

    x  = np.array(x)
    y1 = np.array(acc)
    y2 = np.array(conf)
    y_min = min(y1.min(), y2.min()) -2
    y_max = max(y1.max(), y2.max())

    # plot the bars
    plt.bar(x, +y1-y_min, facecolor='#9999ff', edgecolor='white')
    plt.bar(x, -y2+y_min, facecolor='#ff9999', edgecolor='white')

    # print the values
    for _x,_y1,_y2 in zip(x,y1,y2):
        plt.text(_x+0.4,  _y1-y_min+0.05, '%.1f' % _y1, ha='center', va= 'bottom')
        plt.text(_x+0.4, -_y2+y_min-0.05, '%.1f' % _y2, ha='center', va= 'top')

    # print the text
    for x,m_used in zip(range(nb_models), models_used):
        print str(x)+'_'+str(m_used.nhid)
        if m_used.learning_eps == .25:
            continue
        plt.text(x+1, 0, '%s' % m_used.nhid, ha='center', va= 'bottom')

    plt.text(-.6,  y_max/3, 'Accuracy',   rotation='vertical', verticalalignment='center')
    plt.text(-.6, -y_max/3, 'Confidence', rotation='vertical', verticalalignment='center')


    # set axis
    plt.xlim(-1,nb_models+.5), plt.xticks([])
    plt.ylim(-np.max(y2-y_min)-5,np.max(y1-y_min)+5), plt.yticks([])


    if save_name is not None:
        plt.savefig('./mem/bar_'+save_name+'.png', dpi=100)
    else:
        plt.show()


def plot_epsilon_impact(save_name=None):
    # Prepare data
    tmp = Set('test')
    data = tmp.get_data()

    x = [.0,.05,.1,.15,.2,.25,.3]
    models_used = []
    acc = []
    conf = []
    for eps_adv in x:
        tmp = Model_used(800, eps_adv)
        models_used.append(tmp)
        acc.append(tmp.accuracy(data))
        conf.append(tmp.confidence(data[0]))

    x  = np.array(x)
    y1 = np.array(acc)
    y2 = np.array(conf)
    y_min = min(y1.min(), y2.min()) -2
    y_max = max(y1.max(), y2.max())

    plt.plot(x, y1)
    plt.plot(x, y2, linestyle='dashed')

    if save_name is not None:
        plt.savefig('./mem/bar_'+save_name+'.png', dpi=100)
    else:
        plt.show()


def plot_test_set_impact(bd_name='MNIST', test_set_modif='norm', save_name=None):
    t_s = Set('test', bd_name)

    x = [.0,.05,.1,.15,.2,.25,.3]
    models_used = []
    acc_adv = []
    acc_n_adv = []
    # conf = []
    for eps_test in x:
        for eps_adv in [.0, .25]:
            tmp_model = Model_used(800, eps_adv,'adv', bd_name)
            models_used.append(tmp_model)
            data = t_s.modified_dataset(test_set_modif,eps_test,tmp_model)
            if eps_adv == .0:
                acc_adv.append(tmp_model.accuracy(data))
            else:
                acc_n_adv.append(tmp_model.accuracy(data))
            # conf.append(tmp_model.confidence(data[0]))

    x  = np.array(x)
    y1 = np.array(acc_adv)
    y2 = np.array(acc_n_adv)

    plt.figure()
    plt.plot(x, y1, linestyle ='dashed')
    plt.plot(x, y2)


    if save_name is not None:
        plt.savefig('./mem/bar_'+save_name+'.png', dpi=100)
    else:
        plt.show()


def plot_test_set_impact_other(test_set_modif='norm', save_name=None):
    t_s = Set('test')

    x = [.0,.05,.1,.15,.2,.25,.3]
    models_used = []
    acc_adv = []
    acc_n_adv = []
    acc_n_adv_other = []
    # conf = []
    for eps_test in x:
        for eps_adv in [.0, .25, .99]:
            if eps_adv == .99:
                tmp_model = Model_used(800, .25,'norm')
            else:
                tmp_model = Model_used(800, eps_adv)
            models_used.append(tmp_model)
            data = t_s.modified_dataset(test_set_modif,eps_test,tmp_model)
            if eps_adv == .0:
                acc_adv.append(tmp_model.accuracy(data))
            elif eps_adv == .25:
                acc_n_adv.append(tmp_model.accuracy(data))
            else:
                acc_n_adv_other.append(tmp_model.accuracy(data))
            # conf.append(tmp_model.confidence(data[0]))

    x  = np.array(x)
    y1 = np.array(acc_adv)
    y2 = np.array(acc_n_adv)
    y3 = np.array(acc_n_adv_other)

    plt.figure()
    plt.plot(x, y1, linestyle ='dashed')
    plt.plot(x, y2, linestyle ='dashed')
    plt.plot(x, y3)


    if save_name is not None:
        plt.savefig('./mem/bar_'+save_name+'.png', dpi=100)
    else:
        plt.show()


def plot_weights(nhid, save_name=None):


    plt.figure()
    for plot_row, eps_adv in enumerate([.0, .25]):
        m_used = Model_used(nhid, eps_adv)
        
        nb_plots = min(nhid ,5)
        for i in range(nb_plots):
            plt.subplot(2, nb_plots, i+1+plot_row*nb_plots)
            
            w1_1 = m_used.model.layers[0].get_weights()[:,randint(0,nhid-1)]
            w_shape = int(math.sqrt(w1_1.shape[0]))
            plt.imshow(w1_1.reshape((w_shape,w_shape)), cmap = cm.Greys_r)
            plt.axis('off')

    if save_name is not None:
        plt.savefig('./mem/bar_'+save_name+"_"+str(nhid)+'.png', dpi=100, bbox_inches='tight')
    else:
        plt.show()


def plot_weights_class(nhid, bd_name='MNIST', eps_advs=[.0, .25], save_name=None):

    plt.figure()
    for plot_row, eps_adv in enumerate(eps_advs):
        m_used = Model_used(nhid, eps_adv, bd_name)
        

        nb_plots = 5
        if bd_name=="MNIST_bin": nb_plots = 2   
        for i in range(nb_plots):
            print "------"
            print nhid,"/",eps_adv
            print i+1+(plot_row)*nb_plots
            plt.subplot(len(eps_advs), nb_plots, i+1+(plot_row)*nb_plots)
            w1_1 = m_used.get_weights(0) # size : (in, nhid)
            w2_1 = m_used.get_weights(1)[:,i] # size : (nhid, 1)
            w_shape = int(math.sqrt(w1_1.shape[0]))
            tmp = w1_1.dot(w2_1 + m_used.get_biases(0))
            imgplot = plt.imshow(tmp.reshape((w_shape,w_shape)), cmap = cm.Greys_r)
            plt.axis('off')
            plt.xlabel(str(i))
            imgplot.set_clim(-.5,.5)



    if save_name is not None:
        plt.savefig('./mem/bar_'+save_name+"_"+str(nhid)+'.png', dpi=100, bbox_inches='tight')
    else:
        plt.show()

 
def plot_testset_noisy(noise='adv', use_norm=False, save_name=None):
    if use_norm is False:
        model_used = Model_used(800, .25)
    else :
        model_used = Model_used(800, .25,'norm')
    t_set = Set('test')

    plt.figure()
    for i, eps_noise in  enumerate([.0, .1, .2, .3]):
        data = t_set.modified_dataset(noise, eps_noise, model_used)
        x = data[0][1,:]
        x.shape
        x_shape = int(math.sqrt(x.shape[0]))
        plt.subplot(1,4,i+1)
        plt.imshow(x.reshape((x_shape,x_shape)), cmap = cm.Greys_r)
        plt.axis('off')
        plt.xlabel(eps_noise)
        if i == 0:
            plt.ylabel(noise)

    if save_name is not None:
        plt.savefig('./mem/bar_'+save_name+'.png', dpi=100, bbox_inches='tight')
    else:
        plt.show()


def plot_orig_testset(db_name='MNIST', save_name=None):
    
    t_set = Set('test', db_name)

    plt.figure()
    for i in range(7):
        data = t_set.get_data()
        x = data[0][i,:]
        x_shape = int(math.sqrt(x.shape[0]))
        plt.subplot(1,7,i+1)
        plt.axis('off')
        plt.imshow(x.reshape((x_shape,x_shape)), cmap = cm.Greys_r)
        

    if save_name is not None:
        plt.savefig('./mem/bar_'+save_name+'.png', dpi=100, bbox_inches='tight')
    else:
        plt.show()


plot_orig_testset('MNIST', 'MNIST_orig')

# test_s = Set('test', 'MNIST_bin')
# # for nhid in [2,5,25,100,200,800,1200,1400,1800,2000]:
# for nhid in [400]:
#     plot_weights_class(nhid, 'MNIST_bin', [.0,.05,.1,.15,.2,.25])
#     # print nhid
#     # for eps_adv in [.0,.25]:
#     #     m_used = Model_used(nhid, eps_adv, 'MNIST_bin')
#     #     print m_used.accuracy(test_s.get_data())
#     # print "--"
            

