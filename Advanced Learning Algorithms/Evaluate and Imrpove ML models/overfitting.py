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

def eval_mse(y, yhat):
    m = len(y)
    err = np.sum((y-yhat)**2) / (2*m)
    return err
    
if __name__ == "__main__":
    
    X,y,x_ideal,y_ideal = gen_data(18,2,0.7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    degree = 10
    lmodel = lin_model(degree)
    lmodel.fit(X_train, y_train)
    yhat_train = lmodel.predict(X_train)
    err_train = lmodel.mse(y_train, yhat_train)
    yhat_test = lmodel.predict(X_test)
    err_test = lmodel.mse(y_test, yhat_test)
    print(f"training err {err_train:0.2f}, test err {err_test:0.2f}")
    x = np.linspace(0,int(X.max()),100)
    y_pred = lmodel.predict(x).reshape(-1,1)
    plt_train_test(X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree)
