
import numpy as np


class Logit(object):
    """
    logistic regression using gradient descent!!
    takes three arguments: alpha (learning rate), number of iterations for SGD, and verbose if you want to see output
    """
    def __init__(self, alpha, iterations, verbose):
        self.alpha = alpha
        self.iterations = iterations
        self.verbose = verbose
        self.theta = None
        self.mean = []
        self.std = []

    def sigmoid(self, x):
        '''
        :param x: input vector
        :return: sigmoided vector
        '''
        return 1 / 1(1 + np.exp(-x))

    def gradient_descent(self, X, y):
        """
        Search algorithm - loops over theta and updates to
        take steps in direction of steepest decrease of J.
        :input x: must be numpy array
        :input y: must be numpy vector of 0 and 1
        :return: value of theta that minimizes J(theta) and J_history
        """
        num_examples, num_features = np.shape(X)

        # initialize theta to 1
        self.theta = np.ones(num_features)

        for i in range(self.iterations):
            # difference between hypothesis and actual
            error = self.sigmoid(np.dot(X, self.theta)) - y
            # sum of squares cost
            cost = np.sum(error**2) / (2 * num_examples)
            # calculate average gradient for each row
            gradient = np.dot(X.transpose(), error) / num_examples
            # update the coefficients (theta)
            self.theta = self.theta - self.alpha * gradient

            if i % 5000 == 0 and self.verbose == True:
                print 'iteration:', i
                print 'theta:', self.theta
                print 'cost:', cost

        return self.theta

    def transform(self, data):
        """
        Calculate mean and standard deviation of data
        Transform data by subtracting by mean and
        dividing by std
        :param data: data file
        :return: transformed data
        """

        # transform
        X_norm = data
        for i in range(data.shape[1]):
            mean = np.mean(data[:,i])
            std = np.std(data[:,i])
            self.mean.append(mean)
            self.std.append(std)
            X_norm[:,i] = (X_norm[:,i] - mean) / std

        return X_norm

    def predict(self, X, labels = True):
        """
        Make linear prediction based on cost and gradient descent
        :param X: new data to make predictions on
        :return: return prediction
        """
        num_examples = X.size
        prediction = 0
        for value in range(num_examples):
            prediction = prediction + X[value] * self.theta[value]

        prediction = self.sigmoid(prediction)

        if labels:
            if prediction > 0.5:
                prediction = 1
            else:
                prediction = 0

        return prediction
