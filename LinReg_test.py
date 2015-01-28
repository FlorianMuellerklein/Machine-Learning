import numpy as np
from LinearRegression import LinReg

# initialize linear regression parameters
iterations = 20000
alpha = 0.0001

# plot the data with seaborn (add this later)

linearReg = LinReg(alpha = alpha, iterations = iterations)

# load the example data stolen from 'http://aimotion.blogspot.com/2011/10/machine-learning-with-python-linear.html'
data = np.loadtxt('Data/denver.csv', delimiter = ',')
X = data[:, 0:]
y = data[:, 0]
#data, X, y = linearReg.import_data()

# transform data
X = linearReg.transform(X)

# fit the linear reg
linearReg.gradient_descent(X = X, y = y)

# make a predictions with X = 3.5
print 'prediction:', linearReg.predict(X[5, :])
print 'real value:', y[5]

