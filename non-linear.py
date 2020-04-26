import sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA


x = np.load('x.npy')
y= np.load('y.npy')

x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_test = x[math.floor(len(x) * 0.8): , :]
y_test = y[math.floor(len(y) * 0.8): , :]


dim = 12 * 8 + 1

learning_rate = 10
iter_time = 5000

eps = 0.0000000001
lambda_w1=0.06
lambda_w2=50
w1 = np.zeros([dim, 1])
w2 = np.zeros([dim, 1])
adagrad_1 = np.zeros([dim, 1])
adagrad_2 = np.zeros([dim, 1])
loss_list=[]
power_tmp=np.power(x_train_set,2)

for t in range(iter_time):

    reg_w1 = LA.norm(w1)**2

    reg_w2 = LA.norm(w2)**2

    loss=np.sum(np.power(np.dot(x_train_set, w1)+np.dot(power_tmp, w2)-y_train_set, 2))

    reg_loss=lambda_w1*reg_w1+lambda_w2*reg_w2

    rmse=np.sqrt((loss+reg_loss)/(0.8*(471*12)))

    loss_list.append(rmse)

    if(t%100==0):
        print(str(t) + ":" + str(rmse))
    gradient_1 = 2 * np.dot(x_train_set.transpose(), \
    np.dot(x_train_set, w1)+np.dot(power_tmp, w2) - y_train_set)+2*lambda_w1*w1  #dim*1
    adagrad_1 += gradient_1 ** 2
    w1 = w1 - learning_rate * gradient_1 / np.sqrt(adagrad_1 + eps)

    gradient_2=2 * np.dot(power_tmp.transpose(), \
    np.dot(x_train_set, w1)+np.dot(power_tmp, w2) - y_train_set)+2*lambda_w2*w2
    adagrad_2 += gradient_2 ** 2
    w2 = w2 - learning_rate * gradient_2 / np.sqrt(adagrad_2 + eps)

print(str(iter_time) + ":" + str(loss_list[-1]))



ans_y=np.dot(x_test,w1)+np.dot(np.power(x_test,2), w2)
loss=np.sqrt(np.sum(np.power(ans_y - y_test, 2))/(0.2*(471*12)))
print("final loss:  ", loss)



