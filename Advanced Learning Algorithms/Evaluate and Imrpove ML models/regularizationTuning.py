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
    # Generate  data
    X,y, x_ideal,y_ideal = gen_data(40, 5, 0.7)
    #split the data using sklearn routine
    X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.40, random_state=1)
    X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.50, random_state=1)

    lambda_range = np.array([0.0, 1e-6, 1e-5, 1e-4,1e-3,1e-2, 1e-1,1,10,100])
    num_steps = len(lambda_range)
    degree = 10
    err_train = np.zeros(num_steps)
    err_cv = np.zeros(num_steps)
    x = np.linspace(0,int(X.max()),100)
    y_pred = np.zeros((100,num_steps))

    for i in range(num_steps):
        lambda_ = lambda_range[i]
        lmodel = lin_model(degree, regularization=True, lambda_=lambda_)
        lmodel.fit(X_train, y_train)
        yhat = lmodel.predict(X_train)
        err_train[i] = lmodel.mse(y_train, yhat)
        yhat = lmodel.predict(X_cv)
        err_cv[i] = lmodel.mse(y_cv, yhat)
        y_pred[:,i] = lmodel.predict(x)
    
    optimal_reg_idx = np.argmin(err_cv)
    plt_tune_regularization(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, optimal_reg_idx, lambda_range)
