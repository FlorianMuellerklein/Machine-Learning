
import numpy as np

from BackPropagationNN import BackPropNN

def load_data():
    data = np.loadtxt('Data/train_binary.csv', delimiter = ',')
    y = data[:,0]
    y[y == 0] = -1
    data = data[:,1:]   # x data
    data[data > 0] = 1 # scale features to be on or off
    out = []
    
    for i in range(data.shape[0]):
        fart = list((data[i,:].tolist(), [y[i].tolist()]))
        out.append(fart)

        #time.sleep(20)
        
    return out

if __name__ == '__main__':
    X = load_data()
    
    print X[1]

    NN = BackPropNN(784, 5, 1)

    NN.train(X)