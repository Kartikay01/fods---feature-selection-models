import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


energy_data = 'C:/Users/hp/Downloads/fods_2.csv'
df = pd.read_csv(energy_data)

df = (df - df.mean())/df.std()
train_df = df.sample(frac=0.8, random_state=25)
test_df = df.drop(train_df.index)

x_train = train_df.iloc[:,0:26]
y_train = train_df.iloc[:,-1]
x_test = test_df.iloc[: 0:26]
y_test = test_df.iloc[:,-1]

def weights(x,y):
    
    w = np.ones((x.shape[1],1))
    
    product = np.dot(x.T,x)
    inverse_matrix = np.linalg.inv(product)
    prod = np.dot(inverse_matrix,x.T)
    w = np.dot(prod,y)

    return w

def error(X,subset,y):   
    X=X.iloc[:,subset]
    w=weights(X,y)
    n=len(X)
    err=(1/(2*n))*np.sum(np.square(y-np.dot(X,w)))
    return err


subset2 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
finalrmserror = error(test_df, subset2, y_test)
rms_list_backward_training = []
rms_list_2 = []
rms_list_2.append(subset2)
rms_list_2.append(finalrmserror)
for j in range(26):
    toberemoved = -1
    for i in range(26):
        temp_copy = subset2.copy()
        if(i in temp_copy):
            temp_copy.remove(i)
            temprms = error(train_df, temp_copy, y_train)
            temp_list = []
            temp_list.append(temp_copy)
            temp_list.append(temprms)
            rms_list_2.append(temp_list)
            if(temprms>=finalrmserror):
                finalrmserror = temprms
                toberemoved = i
    if(toberemoved!=-1):
        subset2.remove(toberemoved)
        rms_list_backward_training.append(finalrmserror)
    else:
        break

rms_list_2.pop()
print(subset2)
print(rms_list_backward_training)