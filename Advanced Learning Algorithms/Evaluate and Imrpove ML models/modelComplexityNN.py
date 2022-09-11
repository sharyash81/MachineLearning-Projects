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


def eval_cat_err(y, yhat):
    m = len(y)
    incorrect = 0 

    for i in range(m):
        if yhat[i] != y[i]:
            incorrect+=1

    cerr = incorrect / m 
    return cerr

if __name__ == "__main__":

    X, y, centers, classes, std = gen_blobs()
    X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.50, random_state=1)
    X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.20, random_state=1)

    # complex model
    tf.random.set_seed(1234)
    complex_model = Sequential(
        [
            Dense(units=120, activation="relu", name="layer1"),
            Dense(units=40, activation="relu", name="layer2"),
            Dense(units=6, activation="linear", name="layer3")
        ], name="Complex"
    )

    complex_model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(learning_rate=0.01),
    )
    
    complex_model.fit(
        X_train, y_train,
        epochs=1000
    )

    model_predict = lambda Xl: np.argmax(tf.nn.softmax(complex_model.predict(Xl)).numpy(),axis=1)
    training_cerr_complex = eval_cat_err(y_train, model_predict(X_train))
    cv_cerr_complex = eval_cat_err(y_cv, model_predict(X_cv))

    # simple model
    tf.random.set_seed(1234)
    simple_model = Sequential(
        [
            Dense(units=6, activation="relu", name="layer1"),
            Dense(units=6, activation="linear", name="layer2")
        ], name = "Simple"
    )

    simple_model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(learning_rate=0.01),
    )

    simple_model.fit(
        X_train,y_train,
        epochs=1000
    )

    model_predict_s = lambda Xl: np.argmax(tf.nn.softmax(simple_model.predict(Xl)).numpy(),axis=1)
    training_cerr_simple = eval_cat_err(y_train, model_predict_s(X_train))
    cv_cerr_simple = eval_cat_err(y_cv, model_predict_s(X_cv))
    
    # regularized model
    tf.random.set_seed(1234)
    model_r = Sequential(
        [
            Dense(units=120, activation="relu",name="layer1", kernel_regularizer=tf.keras.regularizers.l2(0.1)),
            Dense(units=40, activation="relu", name="layer2", kernel_regularizer=tf.keras.regularizers.l2(0.1)),
            Dense(units=6, activation="linear", name="layer3")
        ], name= "regularizedComplex"
    )   

    model_r.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=0.01),
    )

    model_r.fit(
        X_train, y_train,
        epochs=1000
    )

    model_predict_r = lambda Xl: np.argmax(tf.nn.softmax(model_r.predict(Xl)).numpy(),axis=1)
    training_cerr_reg = eval_cat_err(y_train, model_predict_r(X_train))
    cv_cerr_reg = eval_cat_err(y_cv, model_predict_r(X_cv))


    print(f"categorization error, training, regularized: {training_cerr_reg:0.3f}, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
    print(f"categorization error, cv,       regularized: {cv_cerr_reg:0.3f}, simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )

    plt_all_model(X_train, y_train, X_cv, y_cv, X_test, y_test, model_predict, model_predict_s, model_predict_r, classes, centers, std)
