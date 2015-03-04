import numpy as np

class LinReg(object):
    """
    multivariate linear regression using gradient descent!!
    takes three arguments: alpha (learning rate), number of iterations for SGD, and verbose if you want to see output
    """
    def __init__(self, alpha = 0.01, iterations = 50, verbose = 0):
        self.alpha = alpha
        self.iterations = iterations
        self.verbose = verbose
        self.theta = None
        self.mean = []
        self.std = []

    def fit(self, X, y):
        """
        Search algorithm - loops over theta and updates to
        take steps in direction of steepest decrease of J.
        :return: value of theta that minimizes J(theta) and J_history
        """
        num_examples, num_features = np.shape(X)

        # initialize theta to 1
        self.theta = np.ones(num_features)

        for i in range(self.iterations):
            # make prediction
            predicted = np.dot(X, self.theta)
            # update theta with gradient descent
            self.theta = self.theta - self.alpha / num_examples * np.dot((predicted - y), X)
            # sum of squares cost
            error = predicted - y
            cost = np.sum(error**2) / (2 * num_examples)
            
            if i % 10 == 0 and self.verbose == 1:
                print 'iteration:', i
                print 'theta:', self.theta
                print 'cost:', cost

        return self.theta

    def predict(self, X):
        """
        Make linear prediction based on cost and gradient descent
        :param X: new data to make predictions on
        :return: return prediction
        """
        num_examples, num_features = np.shape(X)
        prediction = []
        for sample in range(num_examples):
            yhat = 0
            for value in range(num_features):
                yhat += X[sample, value] * self.theta[value]
            prediction.append(yhat)
                
        return prediction

def demo():
    # initialize linear regression parameters
    iterations = 2000
    alpha = 0.1

    linearReg = LinReg(alpha = alpha, iterations = iterations, verbose = 1)

    data = np.genfromtxt('Data/blood_pressure.csv', delimiter = ',', skip_header = 1)
    X = data[:, 1:]
    y = data[:, 0]
    
    # scale data
    max = np.amax(X)
    X /= max
    print X
    print y

    # fit the linear reg
    linearReg.fit(X = X, y = y)

    # load testing dataset
    test = np.genfromtxt('Data/blood_pressure.csv', delimiter = ',', skip_header = 1)
    X_test = test[:, 1:]
    y_test = test[:, 0]
    
    max = np.amax(X_test)
    X_test /= max
    print X_test

    predictions = np.array(linearReg.predict(X_test))

    print 'correct: ', y_test
    print 'prediction: ', predictions
    #np.savetxt('logitpreds.csv', prediction, delimiter = ',')

if __name__ == '__main__':
    demo()