import numpy as np


def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y
