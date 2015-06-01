
import numpy as np
import pylearn2
import pylearn2.train
import pylearn2.datasets
import pylearn2.models.mlp
import pylearn2.training_algorithms.learning_rule
import pylearn2.termination_criteria
import pylearn2.training_algorithms.sgd
import costAdv

X = np.array([
	[  0,  0,  1],
	[  0, .1, .9],
	[  0,  1,  1],
	[ .1, .9, .9],
	[  1,  0,  1],
	[ .9, .1, .9]	])

y = np.array([
	[0],
	[0],
	[1],
	[1],
	[2],
	[2]	])

pickle.dump(X,open("my_X.pkl",'wb'))
pickle.dump(y,open("my_y.pkl",'wb'))



dataset_train = pylearn2.datasets.dense_design_matrix.DenseDesignMatrix(X=X ,y=y ,y_labels=3)
dataset_test =  pylearn2.datasets.dense_design_matrix.DenseDesignMatrix(X=X ,y=y ,y_labels=3)

layer1 = pylearn2.models.mlp.RectifiedLinear(layer_name='h0', dim=3, sparse_init=2)
layer2 = pylearn2.models.mlp.Softmax(layer_name='y' , n_classes=3 , irange=0.)
model_obj = pylearn2.models.mlp.MLP(layers=[layer1,layer2]  ,nvis=3)


learning_rule_obj = pylearn2.training_algorithms.learning_rule.Momentum(init_momentum=.5)
termination_criterion_obj = pylearn2.termination_criteria.EpochCounter(max_epochs=10)
cost_obj = costAdv.AdversarialCost(learning_eps=.25)
algo_obj = pylearn2.training_algorithms.sgd.SGD(
				batch_size=2, 
				learning_rate=.01, 
				learning_rule=learning_rule_obj,
				monitoring_dataset = None,
				termination_criterion = termination_criterion_obj,
				cost = cost_obj	)

Train_obj = pylearn2.train.Train(dataset=dataset_train,
				  model= model_obj,
				  algorithm= algo_obj)



# monitoring_dataset: {
#     'train' : *train,
#     'valid' : !obj:pylearn2.datasets.dense_design_matrix.DenseDesignMatrix {
#         X: !pkl: '/home/marc/data/mnist_valid_X.pkl',
#         y: !pkl: '/home/marc/data/mnist_valid_y.pkl',
#         y_labels: 10,
#     },
#     'test' : !obj:pylearn2.datasets.dense_design_matrix.DenseDesignMatrix {
#         X: !pkl: '/home/marc/data/mnist_test_X.pkl',
#         y: !pkl: '/home/marc/data/mnist_test_y.pkl',
#         y_labels: 10,
#     }, },