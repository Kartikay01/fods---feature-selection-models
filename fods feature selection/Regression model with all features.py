import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations_with_replacement
import array as marray
from mpl_toolkits import mplot3d

energy = 'C:/Users/hp/Downloads/fods_2.csv'
df = pd.read_csv(energy)

df = (df - df.mean())/df.std()

#print(df.head())

df=df.sample(frac=1)
train_df = df.sample(frac=0.8, random_state=35)
test_df = df.drop(train_df.index)
#print(train_df.head())

def compute_cost(X, y, w):
    prediction = X@np.transpose(w)
    #print('predictions', prediction[:5])
    error = np.subtract(prediction, y)
    error_sq = np.square(error)
    J = np.sum(error_sq)/(2*len(X))
    return J

def gradient_descent(X, y , w , eta, iterations):
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        prediction = X@(np.transpose(w))
        error = np.subtract(prediction, y)
        sum_delta = (eta/len(X)) * X.transpose()@error
        w = w - sum_delta.transpose()

        cost_history[i] = compute_cost(X ,y, w)

    return w, cost_history

X_train=train_df.iloc[:,0:26].values
Y_train=train_df.iloc[:,-1].values
X_test=test_df.iloc[:,0:26].values
Y_test=test_df.iloc[:,-1].values

X_train=np.hstack((np.ones((len(X_train),1)), X_train))
X_test=np.hstack((np.ones((len(X_test),1)), X_test))

w=np.zeros(X_train.shape[1])
w, cost = gradient_descent(X_train,Y_train,w,0.0001,10000)
print(w)
train_cost= compute_cost(X_train,Y_train,w)
print("The training error is: ", train_cost)
test_cost = compute_cost(X_test,Y_test,w)
print("The testing error is: ", test_cost)