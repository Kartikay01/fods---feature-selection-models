import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

energy_data = 'C:/Users/hp/Downloads/fods_2.csv'
df = pd.read_csv(energy_data)

#df=df.sample(frac=1)
df =(df-df.mean())/df.std()
training_data = df.sample(frac=0.8, random_state=35)
testing_data = df.drop(training_data.index)
#cor = training_data.corr()

#cor=df.corr()
#plt.figure(figsize=(200,200))
#sns.heatmap(cor,annot=True)
#plt.show()


a = abs(df.corr(method='pearson')['Appliances']).sort_values(ascending=False)
print(a)
desc_list = list(df[a[:].index].columns)
print(desc_list)

#train_df = df.sample(frac=0.80, random_state=35)
#test_df = df.drop(train_df.index)

def insert_ones(x):
    ones = np.ones((x.shape[0],1))
    x= np.concatenate((ones,x),axis=1)
    return x

def weights(x,y):
    x = (x-x.min())/(x.max()-x.min())
    y = (y-y.min())/(y.max()-y.min())
    #x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=90,shuffle=True)
    #x_train = insert_ones(x_train)
    #x_test = insert_ones(x_test)
    x= insert_ones(x)
    w = np.ones((x.shape[1],1))
    
    product = np.dot(x.T,x)
    inverse_matrix = np.linalg.inv(product)
    prod = np.dot(inverse_matrix,x.T)
    w = np.dot(prod,y)

    return w

def error(x,y,w):
    x = insert_ones(x)

    n = len(x)
    error = (1/(2*n))*np.sum(np.square(y - np.dot(x,w)))
    return error

train_errors=[]
test_errors=[]
num = []

for i in range(1,26): 
    num.append(i)
    X = training_data[desc_list[1:i+1]]
    y = training_data['Appliances']

    X_test = testing_data[desc_list[1:i+1]]
    y_test = testing_data['Appliances']
    
    w= weights(X,y)
    print(w)

    train_errors.append(error(X,y,w))
    print(error(X,y,w))
    test_errors.append(error(X_test,y_test,w))
    print(error(X_test,y_test,w))


plt.scatter(num,train_errors)
plt.show()

plt.scatter(num,test_errors)
plt.show()











