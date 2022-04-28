import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N_features = 2            # Number of features
N_train = 1000            # Number of train samples
N_test = 10000            # Number of test samples
N_labels = 2              # Number of classes

# Class Priors and assigning labels
priors = [0.5, 0.5] 
train_label = np.zeros((1, N_train))
train_label = (np.random.rand(N_train) >= priors[1]).astype(int)
train_label = np.array([int(-1) if (t == 0) else int(1) for t in train_label])

test_label = np.zeros((1, N_test))
test_label = (np.random.rand(N_test) >= priors[1]).astype(int)
test_label = np.array([int(-1) if (t == 0) else int(1) for t in test_label])

# Assign values to data samples based on labels
X_train = np.zeros(shape = [N_train, N_features])
X_test = np.zeros(shape = [N_test, N_features])

for i in range(N_train):
    if train_label[i] == 1: 
        X_train[i, 0] = 4 * np.cos(np.random.uniform(-np.pi, np.pi))
        X_train[i, 1] = 4 * np.sin(np.random.uniform(-np.pi, np.pi)) 
    
    elif train_label[i] == -1: 
        X_train[i, 0] = 2 * np.cos(np.random.uniform(-np.pi, np.pi))
        X_train[i, 1] = 2 * np.sin(np.random.uniform(-np.pi, np.pi)) 

    X_train[i, :] += np.random.multivariate_normal([0, 0], np.eye(2))

for i in range(N_test):
    if test_label[i] == 1: 
        X_test[i, 0] = 4 * np.cos(np.random.uniform(-np.pi, np.pi))
        X_test[i, 1] = 4 * np.sin(np.random.uniform(-np.pi, np.pi)) 
    
    elif test_label[i] == -1: 
        X_test[i, 0] = 2 * np.cos(np.random.uniform(-np.pi, np.pi))
        X_test[i, 1] = 2 * np.sin(np.random.uniform(-np.pi, np.pi)) 
        
    X_test[i, :] += np.random.multivariate_normal([0, 0], np.eye(2))

# Plot training dataset
plt.scatter(X_test[(test_label == -1), 0], X_test[(test_label == -1), 1], color = "b")
plt.scatter(X_test[(test_label == 1), 0], X_test[(test_label == 1), 1], color = "r")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title('Test Dataset')
plt.show()

# Save both datasets in a CSV file
train_label = np.reshape(train_label, (1000,1))
dataset = pd.DataFrame(np.hstack((X_train, train_label)))
filename = 'SVM_Dtrain_.csv'
dataset.to_csv(filename, index = False)

test_label = np.reshape(test_label, (10000,1))
dataset = pd.DataFrame(np.hstack((X_test, test_label)))
filename = 'SVM_Dtest_.csv'
dataset.to_csv(filename, index = False)