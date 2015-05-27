
# This is the running example of thesis

import numpy as np
import math


# Usual sigmoid function
sig = lambda x : 1/(1+np.exp(-x))  

# Cost is \sum_k (-y_k*ln(P_k))
do_cost = lambda y, p : np.multiply(-y,np.log(p)).sum(axis=0)

def print_matrix(matrix):
	lig,col = matrix.shape
	print '\left( \\begin{matrix}'
	for y in range(lig):
		
		for x in range(col):
			print round(matrix[y,x],3), 
			if x+1 != col: 
				print '\t&',
			else :
				print '\\\\'

	print '\\end{matrix} \\right)'


class NN:

	def __init__(self):
		self.eps = 0.5
		self.eps_adv = .07
		self.delta1 = None
		self.delta2 = None
		self._set_samples()
		self._set_weights()

	def _set_samples(self):
		self.x = np.matrix(
			[[ 0. 	,0.  	,0. 	,0.1 	,1. 	,0.9],
			 [ 0. 	,0.1 	,1. 	,0.9 	,0. 	,0.1],
			 [ 1. 	,0.9 	,1. 	,0.9 	,1. 	,0.9]	])
		self.x = np.matrix([[ 0., 0., 1.]]).T
		self.x = np.matrix([[ 0, 1, 1]]).T

		self.y = np.matrix(
			[[1 ,1 ,0 ,0 ,0 ,0],
			 [0 ,0 ,1 ,1 ,0 ,0],
			 [0 ,0 ,0 ,0 ,1 ,1]  ])
		self.y = np.matrix([[1,0,0]]).T
		self.y = np.matrix([[0,1,0]]).T

	def _set_weights(self):
		# Weights on the 1st layer
		self.w1 = np.matrix([
			[ -10 , -10 ,   10  ],
			[ -10 ,  10 ,  -10  ],
			[  20 ,  10 ,   10  ]  ])
		self.b1 = np.matrix([-13,-13,-13]).T

		# Weights on the 2nd layer
		self.w2 = np.matrix([
			[  5, -5, -5],
			[ -5,  5, -5],
			[ -5, -5,  5] ])
		self.b2 = np.matrix([0,0,0]).T

	def get_adversarial_input(self):
		if self.delta1 is None:
			self.f_prop()
			self.b_prop()

		twist = self.eps_adv * np.sign(self.w1 * self.delta1)
		return self.x + twist

	def f_prop(self, adv=False):

		if adv is True:
			m_input = self.get_adversarial_input()
		else :
			m_input = self.x

		# Forward to prediction
		self.z1 = self.w1.T*m_input + self.b1
		self.a1 = sig(self.z1)
		self.z2 = self.w2.T*sig(self.z1) + self.b2
		self.a2 = sig(self.z2)
		self.p  = self.a2

		# Normalize prediction
		s = self.p.sum(axis=0)
		s = s.repeat(3,axis=0)
		self.p = self.p/s
		return self._cost(self.y,self.p).mean()

	def b_prop(self, adv=False):
		# compute the deltas
		self.delta2 = (self.p-self.y)

		tmp = np.multiply(self.a1,(1-self.a1))
		self.delta1 = np.multiply(tmp,(self.w2.T*self.delta2))

	def _cost(self,y,p):
		return np.multiply(-y,np.log(p)).sum(axis=0)

	def update(self, adv=False):
		self.f_prop(adv=adv)
		self.b_prop(adv=adv)

		# Compute update value
		self.update_w1 = 0
		self.update_w2 = 0
		for in_nb in range(self.a1.shape[1]):
			self.update_w2 += np.kron(self.a1[:,in_nb],self.delta2[:,in_nb].T) / self.a1.shape[1]
		for in_nb in range(m_input.shape[1]):
			self.update_w1 += np.kron(self.x[:,in_nb],self.delta1[:,in_nb].T) / self.x.shape[1]

		# Update
		self.w1 = self.w1 - self.eps * self.update_w1
		self.w2 = self.w2 - self.eps * self.update_w2
		self.b1 = self.b1 - self.eps * self.delta1
		self.b2 = self.b2 - self.eps * self.delta2

	def m_print(self,what=None):
		print "----- ",what," -----"
		if what == None:
			print "You must fill the 'what' parameter"

		elif what == 'w':
			print self.w1
			print self.w2

		elif what == 'a':
			print self.a1
			print self.a2

		elif what == 'z':
			print self.z1
			print self.z2

		elif what == 'x':
			print self.x

		elif what == 'y':
			print self.y
		
		print ""

a = NN()
print a.f_prop()
print a.f_prop(True)

print a.get_adversarial_input()



# [ 0.07, 0.07, .93]
# => 0.0144329069601

# [ 0.07, 0.07, 1.07]
# => 0.0135801644528

# [ 0., 0., 1.]
# => 0.0135010986448