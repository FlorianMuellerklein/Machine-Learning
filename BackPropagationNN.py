
import numpy as np
np.seterr(all = 'ignore')

# sigmoid transfer function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def dsigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

class BackPropNN(object):
    def __init__(self, input, hidden, output):
        """
        back propagation neural network, sets up all of the matricies we'll need
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden
        self.output = output

        # set activation for nodes
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output

        # create randomized weights
        self.wi = np.random.randn(self.input, self.hidden) # weight vector going from input to hidden
        self.wo = np.random.randn(self.hidden, self.output)  # weight vector going from hidden to output

        # create arrays of 0 for momentum
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedforward(self, inputs):
        """
        The feedforward algorithm loops over all the nodes in the hidden layer and
        adds together all the outputs from the input layer * their weights
        the output of each node is the sigmoid function of the sum of all inputs
        which is then passed on to the next layer.
        :param inputs: input data
        :return: updated activation output vector
        """
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs you silly goose!')

        # input activations
        for i in range(self.input -1): # -1 is to avoid the bias
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, N):
        """
        for the output layer
        1. Calculates the difference between output value and target value
        2. Get the derivative (slope) of the sigmoid function in order to determine how much the weights need to change
        3. update the weights for every node based on the learning rate and sig derivative
        for the hidden layer
        1. calculate the sum of the strength of each output link multiplied by how much the target node has to change
        2. get derivative to determine how much weights need to change
        3. change the weights based on learning rate and derivative
        :param targets: y values
        :param N: learning rate
        :param M: momentum
        :return:
        """
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets you silly goose!')

        # calculate error terms for output
        # the delta tell you which direction to change the weights
        output_deltas = [0.0] * self.output
        for k in range(self.output):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        # delta tells you which direction to change the weights
        hidden_deltas = [0.0] * self.hidden
        for j in range(self.output):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[k] = dsigmoid(self.ah[j]) * error

        # update the weights connecting hidden to output
        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += N * change + self.co[j][k]
                self.co[j][k] = change

        # update the weights connecting input to hidden
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] += N * change + self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[1], '->', self.update(p[0]))

    def train(self, patterns, iterations = 300, N = 0.0002):
        # N: learning rate
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedforward(inputs)
                error = self.backPropagate(targets, N)
            if i % 100 == 0:
                print('error %-.5f' % error)

def demo():
    """
    run NN demo on the digit recognition dataset from sklearn
    :return:
    """
    def load_data():
        data = np.loadtxt('Data/sklearn_digits.csv', delimiter = ',')
        y = data[:,0:10]
        #y[y == 0] = -1
        data = data[:,10:]   # x data
        data -= data.min() # scale the data so values are between 0 and 1
        data /= data.max()
        #data[data == 0] = -1 # scale features to be on or off
        out = []
        print data.shape

        for i in range(data.shape[0]):
            fart = list((data[i,:].tolist(), y[i].tolist()))
            out.append(fart)

            #time.sleep(20)

        return out

    X = load_data()

    print X[9]

    NN = BackPropNN(64, 20, 10)

    NN.train(X)

    NN.test(X)

if __name__ == '__main__':
    demo()
