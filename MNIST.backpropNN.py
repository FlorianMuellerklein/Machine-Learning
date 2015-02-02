
import numpy as np

from BackPropagationNN import BackPropNN

def load_data():
    data = np.loadtxt('Data/sklearn_digits.csv', delimiter = ',')
    y = data[:,0:9]
    print y.shape
    data = data[:,10:]   # x data
    data[data > 0] = 1 # scale features to be on or off
    out = []
    print data.shape
    
    for i in range(data.shape[0]):
        fart = list((data[i,:].tolist(), y[i].tolist()))
        out.append(fart)

        #time.sleep(20)
        
    return out

if __name__ == '__main__':
    X = load_data()
    
    print X[1]

    NN = BackPropNN(64, 20, 9)

    NN.train(X)