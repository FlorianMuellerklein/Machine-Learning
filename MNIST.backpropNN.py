
import numpy as np

from BackPropagationNN import BackPropNN

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

if __name__ == '__main__':
    X = load_data()
    
    print X[9]

    NN = BackPropNN(64, 100, 10)

    NN.train(X)
    
    NN.test(X)