import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *
import warnings


def my_dense(a_in, W, b, g):
    units = W.shape[1]
    a_in, b = a_in.reshape(-1,1) , b.reshape(-1,1)
    z = np.dot(W.T,a_in) + b
    a_out = g(z)
    return a_out


def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = my_dense(x, W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return a3


if __name__ == "__main__":
    
    X, y = load_data()
    model = Sequential(
            [
                tf.keras.Input(shape=(400,)),
                Dense(units="25", activation="sigmoid", name="layer1"),
                Dense(units="15", activation="sigmoid", name="layer2"),
                Dense(units="1" , activation="sigmoid", name="layer3")
            ], name= "my_model"
    )
    model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.001),
    )
    model.fit(
            X,y,
            epochs=20
    )

    [layer1, layer2, layer3] = model.layers
    W1_tmp, b1_tmp = layer1.get_weights()
    W2_tmp, b2_tmp = layer2.get_weights()
    W3_tmp, b3_tmp = layer3.get_weights()


    warnings.simplefilter(action='ignore', category=FutureWarning)
    m, n = X.shape
    fig, axes = plt.subplots(8,8, figsize=(8,8))
    fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20,20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Predict using the Neural Network implemented in Numpy
        my_prediction = my_sequential(X[random_index], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
        my_yhat = int(my_prediction >= 0.5)

        # Predict using the Neural Network implemented in Tensorflow
        tf_prediction = model.predict(X[random_index].reshape(1,400))
        tf_yhat = int(tf_prediction >= 0.5)

        # Display the label above the image
        ax.set_title(f"{y[random_index,0]},{tf_yhat},{my_yhat}")
        ax.set_axis_off()
    fig.suptitle("Label, yhat Tensorflow, yhat Numpy", fontsize=16)
    plt.show()

    
