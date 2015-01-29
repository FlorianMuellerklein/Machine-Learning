import numpy as np

class BackPropagationNetwork:
    '''
    back propagation neural network
    '''

    # methods
    layerCount = 0
    shape = None
    weights = []

    # class methods
    def __init__(self, layerSize):
        '''
        initialize the network
        :param layerSize: sets the layer count
        :return:
        '''

        # layer information
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize

        # input/output data from last run
        self._layerInput = []
        self._layerOutput = []

        # create the weight arrays
        for (l1, l2) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale = 0.01, size = (l2, l1 + 1)))

    # run method
    def run(self, input):
        '''
        run the network based on the input data
        :param input:
        :return:
        '''
        lnCases = input.shape[0]

        # clear out previous intermediate value lists
        self._layerInput = []
        self._layerOutput = []

        # run
        for index in range(self.layerCount):
            # determine layer input
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, lnCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, lnCases])]))

        self._layerInput.append(layerInput)
        self._layerOutput.append(self.sigmoid(layerInput))

        return self._layerOutput[-1].T


    # transfer functions
    def sigmoid(self, x, Derivative = False):
        if not Derivative:
            return 1 / (1 + np.exp(-x))
        else:
            out = self.sigmoid(x)
            return out * (1 - out)


# if run as a script, create test object
if __name__ == '__main__':
    bpn = BackPropagationNetwork((2,2,1))
    print bpn.shape
    print bpn.weights

    lvInput = np.array([[0,0], [1,1], [-1, 0.5]])
    lvOutput = bpn.run(lvInput)

    print 'Input: {0}\nOutput: {1}'.format(lvInput, lvOutput)