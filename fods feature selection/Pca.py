import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

energy_data = 'C:/Users/hp/Downloads/fods_2.csv'
df = pd.read_csv(energy_data)

df = (df - df.mean())/df.std()
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

def insert_ones(x):
    ones = np.ones((x.shape[0],1))
    x= np.concatenate((ones,x),axis=1)
    return x

def weights(x,y):
   
    w = np.ones((x.shape[1],1))
    
    product = np.dot(x.T,x)
    inverse_matrix = np.linalg.inv(product)
    prod = np.dot(inverse_matrix,x.T)
    w = np.dot(prod,y)

    return w

def error(x,y,w):
    
    n = len(x)
    error = (1/(2*n))*np.sum(np.square(y - np.dot(x,w)))
    return error

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


scaling=StandardScaler()
 
# Use fit and transform method
#scaling.fit(X_train)
#Scaled_data_train=scaling.transform(X_train)
#Scaled_data_test=scaling.transform(X_test)

train_err=[]
test_err=[]


for i in range(26):
    principal=PCA(n_components=i+1)
    X_train_pca=principal.fit_transform(X_train)
    X_test_pca = principal.transform(X_test)
    
    X_train_pca=pd.DataFrame(X_train_pca)

    w = weights(X_train_pca,y_train)
    tr_error = error(X_train_pca,y_train,w)
    print("Training and testing error for ", (i+1) , " components")
    train_err.append((i+1,tr_error))
    print(tr_error)
    test_er = error(X_test_pca,y_test,w)
    test_err.append((i+1,tr_error))
    print(test_er)

    
print(principal.explained_variance_ratio_.cumsum())

train_err=pd.DataFrame(train_err)
plt.plot(train_err[0],train_err[1])
plt.title("No. of components vs Training error")
plt.show()

test_err=pd.DataFrame(test_err)
plt.plot(test_err[0],test_err[1])
plt.title("No. of components vs Testing error")
plt.show()




