import theano 
import os.path
import numpy as np
import pickle as pkl
import my_preprocessors as preproc
import matplotlib.pyplot as plt

from pylearn2.utils import serial
from pylearn2.datasets import dense_design_matrix
from pylearn2.format.target_format import convert_to_one_hot



test_eps_array = [.0,.1,.2,.25,.3]
test_eps_array = [.0,.05,.1,.15,.2,.25,.3,.35]
costs = [.0,.1,.2,.25,.3]
ds_s = ['adv', 'norm', 'uni']


class Model_used():
	def __init__(self, model_type, learning_eps=.0, load_model=True):
		self.type = model_type;
		self.learning_eps = learning_eps;
		self.name = str(self.type)+"_"+str(self.learning_eps)
		self.nll = []
		self.acc = []
		self.test_set = 'adv'
		if load_model:
			self.load()


	# File of debug print
	def debug_print_file(self):
		#
		return "./mem/out_"+self.name+".txt"


	##########################
	# Load and save
	##########################

	# load the model if present
	def load(self):
		path_model = "./mem/mlp_"+self.name+".pkl"
		if os.path.isfile(path_model):
			self.model = serial.load(path_model)

	# Load the test pickle
	def load_test(self, test_type='adv'):
		self.test_set = test_type
		path = "./mem/test_"+test_type+"_"+self.name+".pkl"
		if os.path.isfile(path):
			obj =  pkl.load(open(path))
			self.nll, self.acc = obj
		else:
			print "[WARNING]: no test_acc or test_nll was loaded"

	# Save result of testing
	def save_test(self):
		self.test_set
		path = "./mem/test_"+self.test_set+"_"+self.name+".pkl"
		obj = (self.nll, self.acc)
		pkl.dump(obj , open(path, 'w'))


	##########################
	# Test procedure
	##########################

	# Test the model with modified datasets
	def test(self, dataset=None, test_type='adv'):
		model = self.model
		self.test_set = test_type

		# Reset acc and nll arrays
		self.nll = []
		self.acc = []

		# Load the original dataset if not given
		if dataset is None :
			X = pkl.load(open("/home/marc/data/mnist_test_X.pkl"))
			y = pkl.load(open("/home/marc/data/mnist_test_y.pkl"))
			dataset = (X,y)
		X,y = dataset

		# Test the model with different adversarial datasets
		for test_eps in test_eps_array:
			# chose preprocessor
			if test_type == 'adv':
				m_preproc = preproc.Adversarial_modif(model, self.learning_eps, test_eps)
			elif test_type == 'norm':
				m_preproc = preproc.Normal_modif('norm', test_eps)
			elif test_type == 'uni':
				m_preproc = preproc.Normal_modif('uni', test_eps)

			# Apply preprocessor
			ds = dense_design_matrix.DenseDesignMatrix(X=X, y=y, y_labels=10)
			ds.apply_preprocessor(
				preprocessor=m_preproc, 
				can_fit=True)

			# Compute the predcition given model and dataset
			y_hat = model.fprop( theano.shared(ds.X, name='x')).eval()
			Y = convert_to_one_hot(ds.y).squeeze()
			cost = model.cost_from_X(   (theano.shared(ds.X),theano.shared(Y))  ).eval()
			self.nll.append(float(cost))

			# Compute accuracy
			cpt = 0
			for i, preds in enumerate(y_hat):
				if y[i] == np.argmax(preds):
					cpt += 1
			accuracy = cpt / 10000. * 100.
			self.acc.append(accuracy)
			self.save_test()


	##########################
	# Draw and print here
	##########################

	# Draw the test graph
	def draw(self, test_type='adv', show=False ):
		# Load acc and nll
		self.load_test(test_type)
		
		# Draw things
		plt.figure()
		x = np.array(test_eps_array)
		y1 = np.array(self.acc)
		y2 = np.array(self.nll)
		plt.plot(x, y1, x, y2)
		plt.axis([0, .3	, 0, 100])
		
		# Save or show plot
		if show is True:
			plt.show()
		else :
			save_path = "./mem/plot_"+test_type+"_"+self.name+".png"
			plt.savefig(save_path)

	# Output the neurons layer sizes
	def print_type(self):
		if self.type == 1:
			return "1200 + 10"
		if self.type == 2:
			return "1200 + 1200 + 10"

	# Print of class
	def __str__(self):
		m_str = "\n=========\n"
		m_str += "Model is : "+str(self.print_type())+"\n"
		m_str += "Learn eps on model was : "+str(self.learning_eps)
		return m_str


class Model_wapper():
	def __init__(self):
		self.a = 1

	def load_all(self):

		# Create data dictionary
		data ={1:{}, 2:{}}
		for model in data:
			data[model] = {.0:{}, .1:{}, .2:{}, .25:{}, .3:{}}
			for cost in data[model]:
				data[model][cost] = {'adv':{}, 'norm':{}, 'unit':{}}

		# Fill data in dictionary
		for model in [1,2]:
			for cost in costs:
				for ds in ds_s:
					l_model = Model_used(model, cost, load_model=False)
					l_model.load_test(ds)
					data[model][cost][ds] = l_model.acc
		
		self.data = data

	def print_given_model_cost(self, model, cost):
		# define x and y_s
		x = test_eps_array
		y_s = []
		for ds in ds_s:
			y_s.append(self.data[model][cost][ds])


		# Draw things
		plt.figure()
		for y in y_s:
			plt.plot(x, y)
		plt.legend(ds_s)
		# plt.axis([0, .3	, 0, 100])
		plt.show()

	def print_given_model_ds(self, model, ds):
		# define x and y_s
		x = test_eps_array
		y_s = []
		for cost in costs:
			y_s.append(self.data[model][cost][ds])

		# Draw things
		plt.figure()
		for y in y_s:
			plt.plot(x, y)
		# plt.axis([0, .3	, 0, 100])
		plt.legend(costs)
		plt.show()


if __name__ == "__main__":
	# 

	# Draw some stuff
	wrapper = Model_wapper()
	wrapper.load_all()
	wrapper.print_given_model_ds(1, 'norm')
	wrapper.print_given_model_ds(1, 'uni')
