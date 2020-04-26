import sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
import csv


#选择了feature，小时

#load the dataset
x = np.load('x.npy')
y= np.load('y.npy')



dim = 12 * 8 + 1 # dimension number
learning_rate = 10
iter_time = 5000
eps = 0.0000000001
lambda_w=0.06


# 80% of the data set for training
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]

# 20% of the data set for validating
x_test = x[math.floor(len(x) * 0.8): , :]
y_test = y[math.floor(len(y) * 0.8): , :]



w = np.zeros([dim, 1]) # parameters for every feature, plus one more bias
adagrad = np.zeros([dim, 1]) # Adagrad for gradient descent
loss_list=[]


for t in range(iter_time):

    reg_w = LA.norm(w)**2 # regularization term

    reg_loss=lambda_w*reg_w # regularization term with the weight

    loss=np.sum(np.power(np.dot(x_train_set, w)-y_train_set, 2))# square error

    rmse=np.sqrt((loss+reg_loss)/(0.8*(471*12)))

    loss_list.append(rmse)

    if(t%100==0):
        print(str(t) + ":" + str(rmse))

    gradient =  np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set)+2*lambda_w*w  

    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
print(str(iter_time) + ":" + str(loss_list[-1]))

ans_y=np.dot(x_test,w)
loss=np.sqrt(np.sum(np.power(ans_y - y_test, 2))/(0.2*(471*12)))
print("final loss:  ", loss)


