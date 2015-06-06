# transform CIFAR-10 to grayscale

import numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Cifar10:
	""" Class to load the CIFAR10 dataset and transform it into a grayscale matrix
	"""
	def __init__(self,path="/home/marc/Data/cifar-10-batches-py/"):
		self.path = path
		self.train_set = {'x':np.ndarray((0,3072)),'y':np.ndarray((0,1), dtype=np.int64)}
		self.valid_set = {'x':np.ndarray((0,3072)),'y':np.ndarray((0,1), dtype=np.int64)}
		self.test_set  = {'x':np.ndarray((0,3072)),'y':np.ndarray((0,1), dtype=np.int64)}

		# load train, valid and test sets
		for i in range(1,4):
			(x,y) = self._load_batch(i)
			self.train_set['x'] = np.append(self.train_set['x'],x, axis=0)
			self.train_set['y'] = np.append(self.train_set['y'],np.array([y]).reshape((-1,1)), axis=0)

		(x,y) = self._load_batch(4)
		self.valid_set['x'] = np.append(self.valid_set['x'],x, axis=0)
		self.valid_set['y'] = np.append(self.valid_set['y'],np.array([y]).reshape((-1,1)), axis=0)
		
		(x,y) = self._load_batch(5)
		self.test_set['x'] = np.append(self.test_set['x'],x, axis=0)
		self.test_set['y'] = np.append(self.test_set['y'],np.array([y]).reshape((-1,1)), axis=0)

	def _load_batch(self, i):
		b_path = self.path + "data_batch_"+str(i)
		b_file = open(b_path,'rb')
		dict0  = pkl.load(b_file)
		b_file.close()
		return dict0['data'], dict0['labels']

	def data_sets(self):
		return zip([self.train_set, self.valid_set, self.test_set],
					["train", "valid", "test"])

	def make_grayscale(self):
		for m_set,tmp in self.data_sets():
			m_set['x'] = m_set['x'].reshape((-1, 3, 1024)).sum(axis=1)/3/255

	def draw_gray_sample(self,sample_id=1):
		sample_data = self.train_set['x'][sample_id,:]
		image = sample_data.reshape((32,32))
		plt.imshow(image, cmap = cm.Greys_r)
		plt.show()

	def save_sets(self):
		for ds,ds_name in self.data_sets():
			x_path = open(self.path+ds_name+"_X.pkl", "wb")
			y_path = open(self.path+ds_name+"_y.pkl", "wb")
			pkl.dump(ds['x'],x_path)
			pkl.dump(ds['y'],y_path)


a = Cifar10()
a.make_grayscale()
a.save_sets()


