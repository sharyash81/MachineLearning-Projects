import numpy as np
import matplotlib.pyplot as plt
from utils import * 
import math
import sys


def sigmoid(z):
    return 1/(1+np.exp(-z))


def compute_f_wb(X, w, b):
    Z = np.dot(X,w) + b
    return sigmoid(Z)


def compute_cost(X, y, w, b, lambda_= 1):
    m = X.shape[0]
    f_wb = compute_f_wb(X,w,b)
    total_cost = (-np.matmul(y.T,np.log(f_wb))-np.matmul(1-y.T,np.log(1-f_wb))) / m
    return total_cost


def compute_gradient(X, y, w, b, lambda_=None):
    m, n = X.shape
    # compute dj_db
    f_wb = compute_f_wb(X, w, b).reshape(m,1)
    tmp = f_wb - y
    dj_db = np.sum(tmp) / m
    # compute dj_dw
    dj_dw = np.dot(X.T,tmp)/m
    return dj_dw, dj_db


def gradient_descent(X, y, w , b , cost_function, gradient_function, alpha, num_iters, lambda_):
    m,n = X.shape
    w = w.reshape(n,1)
    y = y.reshape(m, 1)
    min_cost = sys.maxsize
    crt_flag = True
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost = cost_function(X, y, w, b, lambda_)
        if cost < min_cost:
            min_cost = cost
        elif cost == min_cost:
            break
        else:
            print("linear regression leads to divergence , maybe the alpha is too big")
            crt_flag = False
            break
    if crt_flag:
        return w, b
    else:
        return None

def predict(X, w, b):
    f_wb = compute_f_wb(X, w, b)
    p = np.array(list(map(lambda x: x >= 0.5, f_wb)))
    return p


if __name__ == "__main__":

    np.random.seed(1)
    w = 0.01 * (np.random.rand(2).reshape(2, 1) - 0.5)
    b = -8
    iterations = 10000
    alpha = 0.001
    student_scores, admission_result = load_data("Data/data1.txt")

    # Plot examples
    # plot_data(student_scores, admission_result[:], pos_label="Admitted", neg_label="Not admitted")
    # Set the y-axis label
    # plt.ylabel('Exam 2 score')
    # Set the x-axis label
    # plt.xlabel('Exam 1 score')
    # plt.legend(loc="upper right")
    # plt.show()

    w, b = gradient_descent(student_scores, admission_result, w, b,
                            compute_cost, compute_gradient, alpha, iterations, 0)
    plot_decision_boundary(w, b, student_scores, admission_result)
    plt.show()
