# everything in this script is stolen from http://aimotion.blogspot.com/2011/10/machine-learning-with-python-linear.html
# the only difference is that I wanted to make a class instead of seperate functions

import numpy as np

class LinReg(object):
    '''Classic linear regression
    First try at simple linear regression class in python
    takes five arguments: train_X, train_y, alpha, theta, iterations for SGD
    '''
    def __init__(self, X, y, alpha, theta, iterations):
		self.X = X
		self.y = y
		self.theta = theta
		self.alpha = alpha
		self.iterations = iterations
    
    def compute_cost(self):
		m = self.y.size
		
		predictions = self.X.dot(self.theta).flatten()
		
		sqErrors = (predictions - y) ** 2
		
		return (1.0 / (2 * m)) * sqErrors.sum()
    
    def gradient_descent(self):
        X = self.X
        y = self.y
        m = y.size
        J_history = np.zeros(shape = (self.iterations, 1))
		
        for i in range(self.iterations):
			predictions = X.dot(self.theta).flatten()
			
			errors_x1 = (predictions - y) * X[:,0]
			errors_x2 = (predictions - y) * X[:,1]
			
			self.theta[0][0] = theta[0][0] - self.alpha * (1.0 / m) * errors_x1.sum()
			self.theta[1][0] = theta[1][0] - self.alpha * (1.0 / m) * errors_x1.sum()
			
			J_history[i, 0] = self.compute_cost()
		
        return theta, J_history
		

# initialize linear regression parameters
theta = np.zeros(shape = (2,1))
iterations = 1500
alpha = 0.01

# load the example data stolen from 'http://aimotion.blogspot.com/2011/10/machine-learning-with-python-linear.html'
data = np.loadtxt('ex1data1.txt', delimiter = ',')
X = data[:,0]
y = data[:,1]

# add a column of ones to X (intercept data)
m = y.size
it = np.ones(shape = (m,2))
it[:,1] = X

# plot the data with seaborn (add this later)

# fit the linear reg
linearReg = LinReg(X = it, y = y, alpha = alpha, theta = theta, iterations = iterations)
theta, J_history = linearReg.gradient_descent()

# make a predictions with X = 3.5
# !!!! figure out how to make a .fit and .predict thing like sklearn !!!

print np.array([1,3.5]).dot(theta).flatten()