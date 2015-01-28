# everything in this script is stolen from http://aimotion.blogspot.com/2011/10/machine-learning-with-python-linear.html
# the only difference is that I wanted to make a class instead of seperate functions

import numpy as np

class LinReg(object):
    """
    Classic linear regression
    First try at simple linear regression class in python
    takes two arguments: alpha (learning rate), number of iterations for SGD
    """
    def __init__(self, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations
        self.theta = []
        self.mean = []
        self.std = []
    
    def import_data(self):
        """
        Import data file
        :return: data, X, and y for data transform
        """
        data = np.loadtxt('ex1data1.txt', delimiter = ',')
        X = data[:,0]
        y = data[:,1]
        return data, X, y
    
    def transform(self, data):
        """
        Calculate mean and standard deviation of data
        Transform data by subtracting by mean and
        dividing by std
        :param data: data file
        :param X: first col from data
        :param y: second col from data
        :return: transformed data
        """

        # this does not really make sense if we will have multiple vars
        # but this is the idea!
        
        # transform
        X_norm = data
        for i in range(data.shape[1]):
            mean = np.mean(data[:,i])
            std = np.std(data[:,i])
            self.mean.append(mean)
            self.std.append(std)
            X_norm[:,i] = (X_norm[:,i] - mean) / std

        return X_norm
    
    def compute_cost(self, X, y, theta):
        """
        Calculate mean squared error by subtracting true from predicted
        and dividing by 2
        :return: mean squared error
        """
        m = y.size
        
        predictions = X.dot(theta).flatten()
        
        sqErrors = (predictions - y) ** 2
        
        return (1.0 / (2 * m)) * sqErrors.sum()
    
    def gradient_descent(self, X, y):
        """
        Search algorithm - loops over theta and updates to
        take steps in direction of steepest decrease of J.
        :return: value of theta that minimizes J(theta) and J_history
        """
        m = y.size
        J_history = []

        # add column of 1s to data to fit intercept
        X_int = np.ones(shape = (m,1))
        X_int = np.hstack((X_int, X))
        theta = np.zeros(shape = (X_int.shape[1], 1))

        #i = int(0)
        #tol = 1
        for i in range(self.iterations):
        #while tol > self.tolerance or i > self.iterations:
            predictions = X_int.dot(theta)
            
            theta_size = theta.size
            
            for j in range(theta_size):
                
                temp = X_int[:, j]
                temp.shape = (m, 1)
                
                errors_x = (predictions - y) * temp
                
                theta[j][0] = theta[j][0] - alpha * (1.0 / m) * errors_x.sum()
            
            J_history.append(self.compute_cost(X_int, y, theta))

            if i % 5000 == 0:
                print 'theta:', theta

        self.theta = theta
        #return theta, J_history

    def predict(self, X):
        """
        Make linear prediction based on cost and gradient descent
        :param X: new data to make predictions on
        :return: return prediction
        """
        X_int = np.hstack((1, X))
        prediction = X_int.dot(self.theta).flatten()
        return prediction


# initialize linear regression parameters
iterations = 200000
alpha = 0.0001

# plot the data with seaborn (add this later)

linearReg = LinReg(alpha = alpha, iterations = iterations)

# load the example data stolen from 'http://aimotion.blogspot.com/2011/10/machine-learning-with-python-linear.html'
data = np.loadtxt('ex1data2.txt', delimiter = ',')
X = data[:, :2]
y = data[:, 2]
#data, X, y = linearReg.import_data()

# transform data
X = linearReg.transform(X)
print X[1,]
print X.shape[0]

# fit the linear reg
linearReg.gradient_descent(X = X, y = y)

# make a predictions with X = 3.5
print linearReg.predict(X[1,:])
print y[1]