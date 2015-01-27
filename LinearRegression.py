# everything in this script is stolen from http://aimotion.blogspot.com/2011/10/machine-learning-with-python-linear.html
# the only difference is that I wanted to make a class instead of seperate functions
#
# nothing is stolen if its open sourced!

import numpy as np

class LinReg(object):
	""" Classic linear regression
	First try at simple linear regression class in python
	takes two arguments: alpha (learning rate), number of iterations for SGD
	"""
	def __init__(self, alpha, iterations):
		self.alpha = alpha
		self.iterations = iterations
		self.theta = np.zeros(shape = (2,1))
		# X and y are used in multiple methods so should be in init (self)
		self.X = [];
		self.y = [];

	def compute_cost(self):
		""" Calculate mean squared error by subtracting true from predicted
		and dividing by 2
		:return: mean squared error
		"""
		m = self.y.size

		predictions = self.X.dot(self.theta).flatten()

		sqErrors = (predictions - self.y) ** 2

		return (1.0 / (2 * m)) * sqErrors.sum()

	def gradient_descent(self):
		""" Search algorithm - loops over theta and updates to
		 take steps in direction of steepest decrease of J.
		:return: value of theta that minimizes J(theta) and J_history
		"""
		# why create extra variables? They just take up more space
		#X = self.X
		#y = self.y
		m = self.y.size
		#theta = self.theta
		J_history = np.zeros(shape = (self.iterations, 1))

		for i in range(self.iterations):
			predictions = self.X.dot(self.theta).flatten()

			errors_x1 = (predictions - self.y) * self.X[:,0]
			errors_x2 = (predictions - self.y) * self.X[:,1]

			self.theta[0][0] = self.theta[0][0] - self.alpha * (1.0 / m) * errors_x1.sum()
			self.theta[1][0] = self.theta[1][0] - self.alpha * (1.0 / m) * errors_x2.sum()

			J_history[i, 0] = self.compute_cost()
			if i % 100 == 0:
				print 'theta:', self.theta

		# no need to return a self.theta because it is self (in init)
		return self.theta, J_history

	def predict(self):
		""" Make linear prediction based on cost and gradient descent

		FLORIAN: if a method has (self, X) then you would this below:
		:param X: EXPLAIN WHAT X IS HERE (note: no need to have X because self.x)
		:return: return prediction
		"""
		# predction = self.X.dot(self.theta).flatten()
		prediction = self.X.dot(self.theta).flatten()
		return prediction

# initialize linear regression parameters
iterations = 1500
alpha = 0.001

# load the example data stolen from 'http://aimotion.blogspot.com/2011/10/machine-learning-with-python-linear.html'
data = np.loadtxt('ex1data1.txt', delimiter = ',')
X = data[:,0]
y = data[:,1]

# add a column of ones to X (intercept data)
m = y.size
it = np.ones(shape = (m,2))
it[:,1] = X
#print it

# plot the data with seaborn (add this later)

# fit the linear reg
linearReg = LinReg(alpha = alpha, iterations = iterations)
theta, J_history = linearReg.gradient_descent(X = it, y = y)

# make a predictions with X = 3.5
print linearReg.predict(np.array([1, 3.5]))
