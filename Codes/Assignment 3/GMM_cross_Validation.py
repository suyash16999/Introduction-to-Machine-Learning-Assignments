import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from random import seed
from random import randrange
from scipy.stats import multivariate_normal

#Import Dataset and store features in X
dataset = pd.read_csv('GMM_Dtrain_10000.csv') 
X = dataset.iloc[:, 0:2].values

# Define number of folds and list to store model orders
kf = KFold(n_splits = 10)
Order = []

# Run training routine multiple times to analyze rate of model order selection
for routine in range(1,30):
    Max_likelihood = []
    # Test model orders rangine from 1 to 6
    for order in range(1,7):
        score = 0
        # Perform 10-fold cross validation and compute log-likelihood in each case
        for train_index, test_index in kf.split(X):
            Xtrain = X[train_index]
            Xtest = X[test_index]
            model = GaussianMixture(n_components = order, init_params = 'random', max_iter = 3500, tol = 1e-9, n_init = 2)
            model.fit(Xtrain)
            score += model.score(Xtest)

        # Compute average score after cross validation
        avg_score = score / 10
        # Store average scores in a list
        Max_likelihood.append(avg_score)
    
    print(Max_likelihood)
    # Identify model order with maximum score
    Order.append(np.argmax(Max_likelihood) + 1)

# Plot a histogram showing frequency of model orders selected
plt.hist(Order, range = (1, 7))
plt.xlabel("Model Order")
plt.ylabel("Average Log Likelihood")
plt.title("Plot for Model Order Selection")
plt.show()
