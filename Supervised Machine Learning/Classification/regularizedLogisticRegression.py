import numpy as np
import matplotlib.pyplot as plt
from utils import *
import math
import sys
from logisticRegression import *


def compute_cost_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    # Calls the compute_cost function that you implemented above
    cost_without_reg = compute_cost(X, y, w, b)
    # You need to calculate this value
    reg_cost = 0.
    for j in range(n):
        reg_cost += w[j] ** 2
    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + (lambda_ / (2 * m)) * reg_cost
    return total_cost


def compute_gradient_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    dj_dw, dj_db = compute_gradient(X, y, w, b)
    dj_dw += ((lambda_ / m) * w)
    return dj_dw, dj_db


if __name__ == "__main__":
    student_scores,admission_result = load_data("Data/data2.txt")
    X_mapped = map_feature(student_scores[:, 0], student_scores[:, 1])
    # Initialize fitting parameters

    np.random.seed(1)
    initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
    initial_b = 1.

    # Set regularization parameter lambda_ to 1 (you can try varying this)
    lambda_ = 0.01;
    # Some gradient descent settings
    iterations = 10000
    alpha = 0.01

    w, b = gradient_descent(X_mapped, admission_result, initial_w, initial_b,
                                          compute_cost_reg, compute_gradient_reg,
                                          alpha, iterations, lambda_)

    plot_decision_boundary(w, b, X_mapped, admission_result)
    plt.show()


