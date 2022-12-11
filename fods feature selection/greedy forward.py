import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

energy_data = 'C:/Users/hp/Downloads/fods_2.csv'
df = pd.read_csv(energy_data)

df = (df - df.mean())/df.std()
train_df = df.sample(frac=0.80, random_state=90)
test_df = df.drop(train_df.index)

x_train = train_df.iloc[:,0:26]
y_train = train_df.iloc[:,-1]
x_test = test_df.iloc[: 0:26]
y_test = test_df.iloc[:,-1]

#subset = [0,3,5]
#x = df.iloc[:,subset]
#print(x)

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

subset1 = []
rms_list_forward_training = []
rms_list_1 = []
finalrmserror = error(train_df, [0], y_train)+1
for j in range(25):
    tobadded = -1
    for i in range(25): 
        temp_copy = subset1.copy()
        if (i not in temp_copy):
            temp_copy.append(i)
            temprms = error(train_df, temp_copy, y_train)
            temp_list = []
            temp_list.append(temp_copy)
            temp_list.append(temprms)
            rms_list_1.append(temp_list)
            if(temprms<=finalrmserror):
                finalrmserror = temprms
                tobeadded = i
    if(tobeadded!=-1):
        subset1.append(tobeadded)
        rms_list_forward_training.append(finalrmserror)

   
print(subset1)
print(rms_list_forward_training)