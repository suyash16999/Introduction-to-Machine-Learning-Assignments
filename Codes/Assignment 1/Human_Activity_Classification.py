import random
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import Dataset
df = pd.read_csv('Introduction-to-Machine-Learning-Assignments/Datasets/train.csv')
Data = df.to_numpy()

# Identifying labels and size of dataset
N = Data.shape[0]          
Y = pd.read_csv('Introduction-to-Machine-Learning-Assignments/Datasets/UCI HAR Dataset/train/y_train.txt')
label = np.squeeze(Y.to_numpy())

# Normalizing data to apply PCA
Data = Data[:, 0:-2]
sc = StandardScaler()
Data = sc.fit_transform(Data)

# Reducing dimensions to obtain 10 principal components
pca = PCA(n_components = 10)
Data = pca.fit_transform(Data)

N_labels = 6            # Number of labels 
N_features = 10         # Number of features

# Compute Mean Vectors and Covariance matrices
mean_matrix = np.zeros(shape = [N_labels, N_features])
covariance_matrix = np.zeros(shape = [N_labels, N_features, N_features])

for i in range(0, N_labels):
    mean_matrix[i, :] = np.mean(Data[(label == i + 1), :], axis = 0)
    covariance_matrix[i, :, :] = np.cov(Data[(label == i + 1), :], rowvar = False)
    covariance_matrix[i, :, :] += (0.00001) * ((np.trace(covariance_matrix[i,:,:]))/LA.matrix_rank(covariance_matrix[i,:,:])) * np.eye(10)
    #Check if covariance matrices are ill-conditioned
    #print(LA.cond(covariance_matrix[i,:,:]))

# Assign 0-1 loss matrix
loss_matrix = np.ones(shape = [N_labels, N_labels]) - np.eye(N_labels)

# Compute class conditional PDF
P_x_given_L = np.zeros(shape = [N_labels, N])
for i in range(0, N_labels):
    P_x_given_L[i, :] = multivariate_normal.pdf(Data, mean = mean_matrix[i, :], cov = covariance_matrix[i, :,:])

# Estimate class priors based on sample count
priors = np.zeros(shape = [N_labels, 1])
for i in range(0, N_labels):
    priors[i] = (np.size(label[np.where((label == i + 1))])) / N

# Compute Class Posteriors using priors and class conditional PDF
P_x = np.matmul(np.transpose(priors), P_x_given_L)
ClassPosteriors = (P_x_given_L * (np.matlib.repmat(priors, 1, N))) / np.matlib.repmat(P_x, N_labels, 1)

# Evaluate Expected risk and decisions based on minimum risk
ExpectedRisk = np.matmul(loss_matrix, ClassPosteriors)
Decision = np.argmin(ExpectedRisk, axis = 0)
print("Average Expected Risk", np.sum(np.min(ExpectedRisk, axis = 0)) / N)

# Estimate Confusion Matrix
ConfusionMatrix = np.zeros(shape = [N_labels, N_labels])
for d in range(N_labels):
    for l in range(N_labels):
        ConfusionMatrix[d, l] = (np.size(np.where((d == Decision) & (l == label - 1)))) / np.size(np.where(label - 1 == l))

np.set_printoptions(suppress=True)
print(ConfusionMatrix)

# Plot Data Distribution
fig = plt.figure()
ax = plt.axes(projection = "3d")
for i in range(1, N_labels + 1):
    ax.scatter(Data[(label==i),1],Data[(label==i),2],Data[(label==i),3], label=i)
plt.xlabel('X3')
plt.ylabel('X1')
ax.set_zlabel('X2')
ax.legend()
plt.title('Data Distribution')
plt.show()

    




