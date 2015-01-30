import math
import numpy as np


class Logit(object):
    """
    logistic regression using gradient descent!!
    takes three arguments: alpha (learning rate), number of iterations for SGD, and verbose if you want to see output
    """
    def __init__(self, alpha, iterations, verbose, tolerance):
        self.alpha = alpha
        self.iterations = iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.theta = None
        self.mean = []
        self.std = []

    def sigmoid(self, x):
        """
        Typical sigmoid function created from input vector x

        :param x: input vector
        :return: sigmoided vector
        """
        # typical sigmoid py line, seems to get errors with arrays
        return 1 / (1 + np.exp(-x))

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
            # make predictions
            predicted = self.sigmoid(np.dot(X, self.theta))
            # update theta with gradient descent
            self.theta = self.theta - self.alpha / num_examples * np.dot((predicted - y), X)
            # sum of squares cost
            error = predicted - y
            cost = np.sum(error**2) / (2 * num_examples)

            if i % 5000 == 0 and self.verbose == True:
                print 'iteration:', i
                print 'theta:', self.theta
                print 'cost:', cost

            if cost < self.tolerance:
                return self.theta
                break

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

        X_int = np.ones(shape =(X_norm.shape[0],1))
        X_norm = np.hstack((X_int, X_norm))

        return X_norm

    def predict(self, X, labels):
        """
        Make linear prediction based on cost and gradient descent

        :param X: new data to make predictions on
        :param labels: boolean
        :return: return prediction
        """
        num_examples = X.size
        prediction = 0
        for value in range(num_examples):
            prediction = prediction + X[value] * self.theta[value]

        prediction = self.sigmoid(prediction)

        if labels:
            if prediction > 0.5:
                prediction = int(1)
            else:
                prediction = int(0)

        return prediction
