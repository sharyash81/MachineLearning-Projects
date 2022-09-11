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

    X, y, centers, classes, std = gen_blobs()
    X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.50, random_state=1)
    X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.20, random_state=1)
    tf.random.set_seed(1234)
    lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    models=[None] * len(lambdas)
    for i in range(len(lambdas)):
        lambda_ = lambdas[i]
        models[i] =  Sequential(
            [
                Dense(120, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
                Dense(40, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
                Dense(classes, activation = 'linear')
            ]
        )
        models[i].compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(0.01),
        )

        models[i].fit(
            X_train,y_train,
            epochs=1000
        )
        print(f"Finished lambda = {lambda_}")

    plot_iterate(lambdas, models, X_train, y_train, X_cv, y_cv)
