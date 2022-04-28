import random
import numpy as np
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=np.inf)

# Initialize the data distribution
N_labels = 4
N = 100000
priors = np.array([[0.25,0.25,0.25,0.25]])

# Create mean and covariance matrices 
mean_matrix = np.zeros((4,3))
mean_matrix[0, :] = [0, 0, 0] 
mean_matrix[1, :] = [0, 0, 3]
mean_matrix[2, :] = [0, 3, 0]   
mean_matrix[3, :] = [3, 0, 0]         
covariance_matrix = np.zeros((4, 3, 3))
covariance_matrix[0,:,:] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
covariance_matrix[1,:,:] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
covariance_matrix[2,:,:] = np.array([[1, 0, -3], [0, .5, 0], [-3, 0, 15]])
covariance_matrix[3,:,:] = np.array([[8, 0, 0], [0, 1, 0], [3, 0, 15]])

# Import dataset
dataset = pd.read_csv('Dtest_100000.csv')
# Split features and labels
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values

# Define 0-1 Loss Matrix
loss_matrix = np.ones(shape = [N_labels, N_labels]) - np.eye(N_labels)

# Compute Class conditional PDF
P_x_given_L = np.zeros(shape = [N_labels, N])
for i in range(N_labels):
    P_x_given_L[i, :] = multivariate_normal.pdf(X,mean = mean_matrix[i, :], cov = covariance_matrix[i, :,:])

# Compute Class Posteriors using priors and class conditional PDF
P_x = np.matmul(priors, P_x_given_L)
ClassPosteriors = (P_x_given_L * (np.matlib.repmat(np.transpose(priors), 1, N))) / np.matlib.repmat(P_x, N_labels, 1)

# Evaluate Expected risk and decisions based on minimum risk
ExpectedRisk = np.matmul(loss_matrix, ClassPosteriors)
Decision = np.argmin(ExpectedRisk, axis = 0)
print("Average Expected Risk", np.sum(np.min(ExpectedRisk, axis = 0)) / N)

# Estimate Confusion Matrix
ConfusionMatrix = np.zeros(shape = [N_labels, N_labels])

for d in range(N_labels):
    for l in range(N_labels):
        ConfusionMatrix[d, l] = (np.size(np.where((d == Decision) & (l == y)))) / np.size(np.where(y == l))

print("Confusion Matrix-")
print(ConfusionMatrix)

