import random
import numpy as np

class Perceptron(object):
    '''
    simple feed forward perceptron
    no bias yet, will add later
    '''
    def __init__(self, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations
        # initialize weights
        self.w = None

    def response(self, X):
        '''
        perceptron response
        :param X: input vector
        :return: perceptron out
        '''
        y = np.dot(X, self.w)
        # in this super simple example perceptron will return 1 for positive guess and -1 for neg
        if y >= 0:
            return 1
        else:
            return 0

    def updateWeight(self, X, error):
        '''
        update the vector of input weights
        :param X: input data
        :param error: prediction != true
        :return: updated weight vector
        '''
        self.w = self.w + self.alpha * np.dot(error, X)

    def train(self, X, y):
        '''
        trains perceptron on vector data by looping each row and updating the weight vector
        :param X: input data
        :param y: correct y value
        :return: updated parameters
        '''
        num_examples, num_features = np.shape(X)
        # initialize weight vector
        self.w = np.random.rand(num_features)

        for i in range(self.iterations):
            for j in range(X.shape[0]):
                prediction = self.response(X[j])
                if prediction != y[j]:
                    error = y[j] - prediction
                    self.updateWeight(X[j], error)