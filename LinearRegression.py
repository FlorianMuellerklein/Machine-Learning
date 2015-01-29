# this script is based on http://aimotion.blogspot.com/2011/10/machine-learning-with-python-linear.html
# I wanted to make a generalized form that could be used in any linear regression problem

import numpy as np

class LinReg(object):
    """
    multivariate linear regression using gradient descent!!
    takes two arguments: alpha (learning rate), number of iterations for SGD, and verbose if you want to see output
    """
    def __init__(self, alpha, iterations, verbose):
        self.alpha = alpha
        self.iterations = iterations
        self.verbose = verbose
        self.theta = None
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
        
        sqErrors = (predictions - y)
        
        return (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)
    
    def gradient_descent(self, X, y):
        """
        Search algorithm - loops over theta and updates to
        take steps in direction of steepest decrease of J.
        :return: value of theta that minimizes J(theta) and J_history
        """
        num_examples, num_features = np.shape(X)

        # initialize theta to 1
        self.theta = np.ones(num_features)

        for i in range(self.iterations):
            # difference between hypothesis and actual
            error = np.dot(X, self.theta) - y
            # sum of squares cost
            cost = np.sum(error**2) / (2 * num_examples)
            # calculate average gradient for each row
            gradient = np.dot(X.transpose(), error) / num_examples
            # update the coefficients (theta)
            self.theta = self.theta - self.alpha * gradient

            if i % 5000 == 0 and self.verbose == True:
                print 'iteration:', i
                print 'theta:', self.theta

        return self.theta
        #return theta, J_history

    def predict(self, X):
        """
        Make linear prediction based on cost and gradient descent
        :param X: new data to make predictions on
        :return: return prediction
        """
        num_examples = X.size
        prediction = 0
        for value in range(num_examples):
            prediction = prediction + X[value] * self.theta[value]

        return prediction

