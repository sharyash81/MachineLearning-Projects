import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt
from autils import * 


def my_softmax(z):

    ez = np.exp(z)
    a = ez / np.sum(ez)
    return a


if __name__ == "__main__":

    X, y = load_data()

    tf.random.set_seed(1234)
    model = Sequential (
            [
                Dense(units=25, activation='relu', name="layer1"),
                Dense(units=15, activation='relu', name="layer2"),
                Dense(units=10, activation='linear', name="layer3")

            ], name = "my_model"
    )

    model.compile(
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    )

    history=model.fit (
            X,y,
            epochs=60
    )
    # plot_loss_tf(history)
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    m, n = X.shape
    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]
    widgvis(fig)
    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20,20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Predict using the Neural Network
        prediction = model.predict(X[random_index].reshape(1,400))
        prediction_p = tf.nn.softmax(prediction)
        yhat = np.argmax(prediction_p)

        # Display the label above the image
        ax.set_title(f"{y[random_index,0]},{yhat}",fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Label, yhat Tensorflow, yhat Numpy", fontsize=14)
    plt.show()
