import numpy as np
from LogisticClassifier import Logit

##########################################################
########## Test file for Logistic Classifier #############
##########################################################

# initialize linear regression parameters
iterations = 100000
alpha = 0.01

# plot the data with seaborn (add this later)

lgit = Logit(alpha = alpha, iterations = iterations, verbose = True, tolerance = 0.02)

# load the example data stolen from 'http://aimotion.blogspot.com/2011/10/machine-learning-with-python-linear.html'
data = np.loadtxt('Data/heart.txt', delimiter = ',')
X = data[:, 1:]
y = data[:, 0]

##########################################################
################### DEBUG BABY! ##########################

#import pdb
#pdb.set_trace()
# creates breakpoint for manually interaction dawg

# transform data
#X = lgit.transform(X)
#print X[1,:]

# fit the linear reg
lgit.gradient_descent(X = X, y = y)

# load testing dataset
test = np.loadtxt('Data/heart_test.txt', delimiter = ',')
X_test = test[:, 1:]
y_test = test[:, 0]

# transform testing data
#X_test = lgit.transform(X_test)
#print X_test[1,:]

# make a predictions
prediction = np.zeros(shape = (y_test.size, 2))
correct = 0
for i in range(y_test.size):
    prediction[i,0] = lgit.predict(X_test[i, :], labels = True)
    prediction[i, 1] = y_test[i]
    if prediction[i, 0] == prediction[i, 1]:
        correct += 1

print 'correct: ', correct
np.savetxt('logitpreds.csv', prediction, delimiter = ',')