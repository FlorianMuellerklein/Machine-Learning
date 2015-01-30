import numpy as np

class Perceptron(object):
    '''
    simple feed forward perceptron
    no bias yet, will add later
    '''
    def __init__(self, alpha, iteration)
        self.alpha = alpha
        self.iteration = iteration
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
            return -1

    def updateWeight(self, X, error):
        '''
        update the vector of input weights
        :param X: input data
        :param error: prediction != true
        :return: updated weight vector
        '''

        for i in range(self.w.size):
            self.w = self.alpha * error * X[i]

    def train(self, X, y):
        '''
        trains perceptron on vector data by looping each row and updating the weight vector
        :param X: input data
        :param y: correct y value
        :return: updated parameters
        '''

        # initialize weight vector
        self.w = np.randint(-1,1, shape =(X.shape[1], 1))

        for i in nrange(self.iteration):
            prediction = self.response(X[i,:])
            if prediction != y[i]:
                error = y - prediction
                self.updateWeight(X, error)