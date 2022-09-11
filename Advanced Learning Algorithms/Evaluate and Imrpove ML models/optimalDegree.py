import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from utils import *

if __name__ == "__main__":
    X,y, x_ideal,y_ideal = gen_data(40, 5, 0.7)
    X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.40, random_state=1)
    X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.50,random_state=1)
    max_degree = 9
    err_train = np.zeros(max_degree)
    err_cv = np.zeros(max_degree)
    x = np.linspace(0,int(X.max()),100)
    y_pred = np.zeros((100,max_degree))
    
    for degree in range(max_degree):
        lmodel = lin_model(degree+1)
        lmodel.fit(X_train, y_train)
        yhat = lmodel.predict(X_train)
        err_train[degree] = lmodel.mse(y_train, yhat)
        yhat = lmodel.predict(X_cv)
        err_cv[degree] = lmodel.mse(y_cv, yhat)
        y_pred[:,degree] = lmodel.predict(x)

    optimal_degree = np.argmin(err_cv)+1 
    plt_optimal_degree(X_train, y_train, X_cv, y_cv, X_test, y_test, x, y_pred, x_ideal, y_ideal,
                   err_train, err_cv, optimal_degree, max_degree)
