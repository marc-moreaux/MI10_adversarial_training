import math
import theano 
import os.path
import numpy as np
import pickle as pkl
import my_preprocessors as preproc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage.interpolation

from pylearn2.utils import serial
from pylearn2.datasets import dense_design_matrix
from pylearn2.format.target_format import convert_to_one_hot

from my_preprocessors import Normal_modif
from my_preprocessors import Adversarial_modif

from random import randint


class Model_used():
    def __init__(self, learning_eps='.25', nb_neurons=800, model_type='adv', db_name="MNIST"):
        self.model_type = model_type;
        self.learning_eps = learning_eps;
        self.nb_neurons = nb_neurons
        self.db_name = db_name

        if db_name == 'CIFAR':
            self.nb_neurons = 2500

        # set name
        self.model_name = str(self.learning_eps)+"_"+str(self.nb_neurons)
        if model_type == 'norm':
            self.model_name = self.model_name+"_norm"
        
        # load model
        self.model = self.load()


    def load(self):
        path_model = "./mem/mlp_"+self.db_name+"_"+self.model_name+".pkl"
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






class Test_set():
    def __init__(self, db_name="MNIST"):
        # load test set
        if db_name == 'MNIST':
            x_path = '/home/marc/data/mnist_test_X.pkl'
            y_path = '/home/marc/data/mnist_test_y.pkl'
        if db_name == 'CIFAR':
            x_path = '/media/marc/SAMSUNG_SD_/data/test_X.pkl'
            y_path = '/media/marc/SAMSUNG_SD_/data/test_y.pkl'

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
        
        elif m_type == 'rot':
            X = ds.X
            for idx, img in enumerate(X):
                img = img.reshape(28,28)
                img = scipy.ndimage.interpolation.rotate(img, eps_to_modif*100, reshape=False)
                X[idx] = img.reshape(28*28)
            ds = dense_design_matrix.DenseDesignMatrix(X=X, y=self.y, y_labels=10)
        
        elif m_type == 'zoom':
            X = ds.X
            for idx, img in enumerate(X):
                img = img.reshape(28,28)
                img = scipy.ndimage.interpolation.zoom(img, eps_to_modif)
                X[idx] = img.reshape(28*28)
            ds = dense_design_matrix.DenseDesignMatrix(X=X, y=self.y, y_labels=10)
        
        elif m_type == 'shift':
            X = ds.X
            for idx, img in enumerate(X):
                img = img.reshape(28,28)
                img = scipy.ndimage.interpolation.shift(img, eps_to_modif*3)
                X[idx] = img.reshape(28*28)
            ds = dense_design_matrix.DenseDesignMatrix(X=X, y=self.y, y_labels=10)

        elif m_type == 'none':
            return ds.X, self.y

        return ds.X, self.y



def plot_neurons_impact(save_name=None):
    # Prepare data
    tmp = Test_set()
    data = tmp.get_data()

    a = Model_used(.0,100)
    a.accuracy(data)

    models_used = []
    acc = []
    conf = []
    for nb_neurons in [2,5,25,50,100,400,800,1200,2000]:
        for eps_adv in [.0, .25]:
            tmp = Model_used(eps_adv,nb_neurons)
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
        print str(x)+'_'+str(m_used.nb_neurons)
        if m_used.learning_eps == .25:
            continue
        plt.text(x+1, 0, '%s' % m_used.nb_neurons, ha='center', va= 'bottom')

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
    tmp = Test_set()
    data = tmp.get_data()

    x = [.0,.05,.1,.15,.2,.25,.3]
    models_used = []
    acc = []
    conf = []
    for eps_adv in x:
        tmp = Model_used(eps_adv,800)
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
    t_s = Test_set(bd_name)

    x = [.0,.05,.1,.15,.2,.25,.3]
    models_used = []
    acc_adv = []
    acc_n_adv = []
    # conf = []
    for eps_test in x:
        for eps_adv in [.0, .25]:
            tmp_model = Model_used(eps_adv,800,'adv', bd_name)
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
    t_s = Test_set()

    x = [.0,.05,.1,.15,.2,.25,.3]
    models_used = []
    acc_adv = []
    acc_n_adv = []
    acc_n_adv_other = []
    # conf = []
    for eps_test in x:
        for eps_adv in [.0, .25, .99]:
            if eps_adv == .99:
                tmp_model = Model_used(.25,800,'norm')
            else:
                tmp_model = Model_used(eps_adv,800)
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


def plot_weights(nb_neurons, save_name=None):


    plt.figure()
    for plot_row, eps_adv in enumerate([.0, .25]):
        m_used = Model_used(eps_adv, nb_neurons)
        
        nb_plots = min(nb_neurons ,5)
        for i in range(nb_plots):
            plt.subplot(2, nb_plots, i+1+plot_row*nb_plots)
            
            w1_1 = m_used.model.layers[0].get_weights()[:,randint(0,nb_neurons-1)]
            w_shape = int(math.sqrt(w1_1.shape[0]))
            plt.imshow(w1_1.reshape((w_shape,w_shape)), cmap = cm.Greys_r)
            plt.axis('off')

    if save_name is not None:
        plt.savefig('./mem/bar_'+save_name+"_"+str(nb_neurons)+'.png', dpi=100, bbox_inches='tight')
    else:
        plt.show()


def plot_weights_class(nb_neurons, save_name=None):

    
    plt.figure()
    for plot_row, eps_adv in enumerate([.0, .25]):
        m_used = Model_used(eps_adv, nb_neurons)
        
        nb_plots = 5
        for i in range(nb_plots):
            i=i+1
            plt.subplot(2, nb_plots, i+(plot_row)*nb_plots)
            w1_1 = m_used.model.layers[0].get_weights() # size : (in, nb_neurons)
            w2_1 = m_used.model.layers[1].get_weights()[:,i] # size : (nb_neurons, 1)
            w_shape = int(math.sqrt(w1_1.shape[0]))
            tmp = w1_1.dot(w2_1)
            plt.imshow(tmp.reshape((w_shape,w_shape)), cmap = cm.Greys_r)
            plt.axis('off')
            plt.xlabel(str(i))

    if save_name is not None:
        plt.savefig('./mem/bar_'+save_name+"_"+str(nb_neurons)+'.png', dpi=100, bbox_inches='tight')
    else:
        plt.show()


def plot_testset_noisy(noise='adv', use_norm=False, save_name=None):
    if use_norm is False:
        model_used = Model_used(.25,800)
    else :
        model_used = Model_used(.25,800,'norm')
    t_set = Test_set()

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
    
    t_set = Test_set(db_name)

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




t_s = Test_set()
x = [.0,.1,.2,.3]
imgs = []
for i,eps_test in enumerate(x):
    plt.subplot(1,4,i+1)
    data = t_s.modified_dataset('rot',eps_test,None)
    imgs.append(np.copy(data[0][1,:]))
    plt.imshow(imgs[i].reshape(28,28), cmap = cm.Greys_r)

plt.savefig('./mem/bar_plot_testset_rot.png', dpi=100, bbox_inches='tight')
plt.show()

plot_test_set_impact('MNIST','rot', save_name='test_set_impact_rot')


# m_used = Model_used(.25,2500,'adv','CIFAR')
# test_s = Test_set('CIFAR')

# print m_used.accuracy(test_s.get_data())



# plot_neurons_impact("neuron_impact")

# plot_epsilon_impact("eps_imapct")

# plot_test_set_impact('MNIST', 'norm',"testset_impact_norm")
# plot_test_set_impact('MNIST', 'adv' ,"testset_impact_adv")
# plot_test_set_impact_other('norm',"testset_impact_2_norm")
# plot_test_set_impact_other('adv' ,"testset_impact_2_adv")

# for nb_neurons in [2,5,100,800,2000]:
#     plot_weights(nb_neurons,"weight")

# for nb_neurons in [2,5,25,50,100,400,800,1200,1400,2000]:
#     plot_weights_class(nb_neurons,"weight_class")

# plot_testset_noisy('adv' , False, "noisy_ts_adv")
# plot_testset_noisy('norm', False, "noisy_ts_norm")
# plot_testset_noisy('adv' , True, "noisy_ts_adv_norm")



# CIFAR 10

# plot_test_set_impact('CIFAR', 'norm' ,"testset_impact_norm_CIFAR")
# plot_orig_testset('CIFAR', 'orig_CIFAR')