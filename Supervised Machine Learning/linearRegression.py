import numpy as np 
import matplotlib.pyplot as plt
import sys

def compute_cost(x,y,w,b) : 
    m = x.shape[0]
    total_cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = ( f_wb - y[i] ) ** 2
        total_cost += cost
    total_cost /= ( 2*m )

    return total_cost

def compute_gradient(x,y,w,b) :
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0 
    
    for i in range(m) :
        f_wb = w * x[i] + b
        dis_error = f_wb - y[i]
        dj_dw += ( dis_error * x[i] )
        dj_db += dis_error

    dj_dw /= m
    dj_db /= m
    return dj_dw , dj_db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters) : 
    
    m = x.shape[0]
    w = w_in 
    b = b_in    
    min_cost = sys.maxsize
    crt_flag = True
    for i in range(num_iters):
        dj_dw,dj_db = gradient_function(x,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost = cost_function(x,y,w,b)
        if cost < min_cost :
            min_cost = cost
        elif cost == min_cost :
            break;
        else :
         print("linear regression leads to divergence , maybe the alpha is too big")
         crt_flag = False
         break;
    if crt_flag :
        return w,b
    else :
        return None 

if __name__ == "__main__":
    
    # number of reported cities 
    city_num = int(input("Enter the number of cities : "))
    # create an numpy array for storing population of each city
    city_pop = np.zeros(city_num)
    # create an numpy array for storing profit of each city
    city_prf = np.zeros(city_num)
    for i in range(city_num):
        city_pop[i] , city_prf[i] = input("Enter the population and priofit of the {}th : ".format(i)).split()

    initial_w = 0 
    initial_b = 0 
    iterations = 1500
    alpha = 0.01
    w,b = gradient_descent(city_pop,city_prf,initial_w,initial_b,compute_cost,compute_gradient,alpha,iterations)
    predicted = np.zeros(city_num)
    for i in range(city_num):
        predicted[i] = w * city_pop[i] + b

    # visualizing the raw data
    figure , axis = plt.subplots(1,2)
    axis[0].scatter(city_pop,city_prf,marker='x',c='r')
    axis[0].set_title("Profits vs. Population per city")
    axis[0].set_xlabel("Population of City in 10ks'")
    axis[0].set_ylabel("Profit in $10k")
    # visualizing the linear regression
    axis[1].plot(city_pop,predicted,c="y")
    axis[1].scatter(city_pop,city_prf,marker="o",c="b")
    axis[1].set_title("Applied linear regression on the left plot")
    axis[1].set_xlabel("Population of City in 10ks'")
    axis[1].set_ylabel("Profit in $10k")
    plt.show()
