import draw_images

import model
import numpy as np
import pickle as pkl

from enum import Enum
from pylearn2.utils import serial
from pylearn2.datasets import dense_design_matrix
from pylearn2.format.target_format import convert_to_one_hot




# Initialize sumup Arrays
sumup_acc = []
sumup_nll = []


# Load the original dataset
X = pkl.load(open("/home/marc/data/mnist_test_X.pkl"))
y = pkl.load(open("/home/marc/data/mnist_test_y.pkl"))


###########################################
### Test over the models of 1 and 2
###########################################
# for model in [1,2]:
# 	for learning_eps in [.0, .1 ,.2, .25, .3]:

# 		# Initialize model used
# 		used_model = model.Model_used(1,learning_eps)
# 		print used_model

# 		# Adversarial test
# 		used_model.test(dataset=(X, y), test_type='adv')
# 		used_model.draw(test_type='adv')
# 		used_model.test(dataset=(X, y), test_type='norm')
# 		used_model.draw(test_type='norm')
# 		used_model.test(dataset=(X, y), test_type='uni')
# 		used_model.draw(test_type='uni')

# 		# Update sumup array
# 		sumup_acc.append(used_model.acc)
# 		sumup_nll.append(used_model.nll)


###########################################
### Test the model 3
###########################################
# Initialize model used
used_model = model.Model_used(3)
print used_model

# Adversarial test
used_model.test(dataset=(X, y), test_type='adv')
used_model.draw(test_type='adv')
used_model.test(dataset=(X, y), test_type='norm')
used_model.draw(test_type='norm')
used_model.test(dataset=(X, y), test_type='uni')
used_model.draw(test_type='uni')

# Update sumup array
sumup_acc.append(used_model.acc)
sumup_nll.append(used_model.nll)


# Print the output
# for latex report
for a,b in zip(sumup_acc,sumup_nll):
	for acc in a:
		if acc < 10:
			print "$0{0:.2f}\%$\t&".format(acc),
		else:
			print "${0:.2f}\%$\t&".format(acc),
	print ""
	for nll in b:
		if nll < 10:
			print "$0{0:.4f}$\t&".format(nll),
		else:
			print "${0:.4f}$\t&".format(nll),
	print ""


