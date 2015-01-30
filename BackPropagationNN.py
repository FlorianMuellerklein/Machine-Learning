import math
import random
import numpy as np

random.seed(0)

def sigmoid(x):
    return math.tanh(x)

# derivative of sigmoid
def dsigmoid(y):
    return 1.0 - y ** 2

class BackPropNN(object):
    def __init__(self, input, hidden, output):
        '''
        back propagation neural network, sets up all of the matricies we'll need
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        '''
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden
        self.output = output

        # set activation for nodes
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output

        # create randomized weights
        self.wi = np.random.rand(self.input, self.hidden)
        self.wo = np.random.rand(self.hidden, self.output)

        # create arrays of 0 for momentum
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def update(self, inputs):
        '''
        update the activation nodes of the output vector
        each individual calculation is the sum of the ainput for each layer multiplied by the weights
        :param inputs: input data
        :return: updated activation output vector
        '''
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs you silly goose!')

        # input activations
        for i in range(self.input -1): # -1 is to avoid the bias
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, N, M):
        '''

        :param targets: y values
        :param N:
        :param M:
        :return:
        '''
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets you silly goose!')

        # calculate error terms for output
        output_deltas = [0.0] * self.output
        for k in range(self.output):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.hidden
        for j in range(self.output):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[k] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def train(self, patterns, iterations = 10000, N = 0.01, M = 0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)

def demo():
    # teach network XOR
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # create a network with two input, two hidden, and one output
    n = BackPropNN(2, 2, 1)
    # train
    n.train(pat)
    #test
    n.test(pat)

if __name__ == '__main__':
    demo()