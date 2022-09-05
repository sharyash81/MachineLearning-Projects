import numpy as np
import matplotlib.pyplot as plt

def load_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    return X, y


def widgvis(fig):
     fig.canvas.toolbar_visible = False
     fig.canvas.header_visible = False
     fig.canvas.footer_visible = False


def plot_loss_tf(history):
    fig,ax = plt.subplots(1,1, figsize = (4,3))
    widgvis(fig)
    ax.plot(history.history['loss'], label='loss')
    ax.set_ylim([0, 2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss (cost)')
    ax.legend()
    ax.grid(True)
    plt.show()
