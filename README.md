# MLP-from-scratch
This is my first attempt at creating a neural net (multi-layered-perceptron) from scratch, only using numpy


```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
```


```python
## Read in the data
df = pd.read_csv('https://raw.githubusercontent.com/CasKirch/MLP-from-scratch/main/churn.csv')
del df['customerID']
del df['MultipleLines'] 
df = df[df.TotalCharges != ' ']
df['TotalCharges'] = df['TotalCharges'].astype(float)
df.shape
```




    (7032, 19)



## Preprocessing


```python
df.head()
ohe_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
              'PaymentMethod']

## OneHotEncoding dependent variable
'''Very ugly but I need it to be coded correctly for backprop to work, for IVs I don't care '''

df = pd.get_dummies(df, columns=['Churn'], drop_first=True)
```


```python
y = df.Churn_Yes
X = df
del X['Churn_Yes']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
```


```python
ohe = OneHotEncoder(sparse=False)
ohe.fit(X_train[ohe_columns])
```




    OneHotEncoder(sparse=False)




```python
feature_array = ohe.transform(X_train[ohe_columns])
feature_labels = ohe.categories_

ohe_data = pd.DataFrame(feature_array, columns=np.concatenate(feature_labels).ravel())
```


```python
## Deleting string columns etc
for i in X_train.columns:
    if i in ohe_columns:
        del X_train[i]
```


```python
## Replacing with ohe variables
ohe_data.reset_index(drop=True, inplace=True)
X_train.reset_index(drop=True, inplace=True)

X_train = pd.concat([X_train, ohe_data], axis=1)
```


```python
## Now same for X_test

feature_array = ohe.transform(X_test[ohe_columns])
ohe_data = pd.DataFrame(feature_array, columns=np.concatenate(feature_labels).ravel())

for i in X_test.columns:
    if i in ohe_columns:
        del X_test[i]
```


```python
## Replacing with ohe variables

ohe_data.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

X_test = pd.concat([X_test, ohe_data], axis=1, join = 'inner')
```

## Creating the model


```python
## We need to work with numpy arrays rather than pandas dataframe
X_train = np.array(X_train)
y_train = np.array(y_train)
```


```python
## Save the dimensions of the data
m, n = X_train.shape
```


```python
## Transpose input data
X_train = X_train.T
y_train = y_train.T
```


```python
## Initialize parameters  --> w1, w2, b1, b2
'''I suspect that our network needs to be deeper (more hidden layers) because our dataset does not have high dimensionality'''
def init_params():
    W1 = np.random.rand(10, 42) ## Create an array with random initialization values [-.5, .5] that fit our model (i.e. 10 x 43)
    b1 = np.random.rand(10,1) - 0.5 # 10x1
    W2 = np.random.rand(1, 10) # 10 x 10 --> Should this be 1 x 10 or 10 x 10 ???! Although shape of A2 still does not change!!
    b2 = np.random.rand(1,1) -0.5 # 10 x 1
    
    return W1, b1, W2, b2

```


```python
## Forward prop
def ReLU(Z):
    '''define a ReLU function using numpy'''
    return np.maximum(Z, 0)

def Sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def forward_prop(W1, b1, W2, b2, X): ## Not sure about the X here
    '''Here we can calculate Z1'''
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = Sigmoid(Z2)
    
    return Z1, A1, Z2, A2 
```


```python
## Back prop
def deriv_ReLU(Z):
    '''Derivative of ReLU is either zero or one (is x>0, ReLU is just linear)'''
    return Z > 0 ## True is interpreted as 1, False as 0

def back_prop(Z1, A1, Z2, A2, W2, Y, X):
    ## Second layer
    dZ2 = A2 - Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m)* np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    
    
    # First layer
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    
    return dW1, db1, dW2, db2 
    
```


```python
## Update params
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    return W1, b1, W2, b2
```


```python
## Gradient_descent 
accuracy = list()
def get_predictions(A2):
    prediction=list()
    for i in A2:
        for j in i: ## Really ugly but ok
            if j > .5:
                prediction.append(1)
            else:
                prediction.append(0)
    return prediction

def get_accuracy(predictions, y):
    return np.sum(predictions == y) / len(y)

def gradient_descent(X, y, iterations, learning_rate):
    # 1. Initiate values
    W1, b1, W2, b2 = init_params()
    
    # 2. For iterations:
    for i in range(iterations):
        
            #a Forward prop
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train) 
            #b back_prop
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, y_train, X_train)
            #c update_params
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        
        accuracy.append(get_accuracy(get_predictions(A2), y_train))
    # 3. Show accuracy for each iteration
        if i % 1000 == 0:
            print("On iteration {w}".format(w=i))
            print('You accuracy is {w}'.format(w=get_accuracy(get_predictions(A2), y_train)))
            accuracy_over_time = list()
            accuracy_over_time.append(get_accuracy(get_predictions(A2), y_train))
    return accuracy                         
    
```


```python
iterations = 10000
acc = gradient_descent(X_train, y_train, iterations, 0.0001)
plt.plot(acc)
```

    On iteration 0
    You accuracy is 0.26801517067003794
    On iteration 1000
    You accuracy is 0.7447850821744627
    On iteration 2000
    You accuracy is 0.7493678887484198
    On iteration 3000
    You accuracy is 0.7586915297092288
    On iteration 4000
    You accuracy is 0.7710176991150443
    On iteration 5000
    You accuracy is 0.7738621997471555
    On iteration 6000
    You accuracy is 0.7779709228824273
    On iteration 7000
    You accuracy is 0.7752844500632111
    On iteration 8000
    You accuracy is 0.7749683944374209
    On iteration 9000
    You accuracy is 0.775126422250316
    
 
