import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def create_datasets():
    Ntrain = [10, 100, 1000, 10000]
    for N in Ntrain:
        data, labels = generateData(N)
        labels = np.squeeze(labels)
        plot(data, labels)
        dataset = pd.DataFrame(np.transpose(np.vstack((data, labels))))
        filename = 'GMM_Dtrain_' + str(N) +'.csv'
        dataset.to_csv(filename, index = False)


def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.2,.3,.35,.15] # priors should be a row vector
    gmmParameters['meanVectors'] = np.zeros((4,2))
    gmmParameters['meanVectors'][0, :] = [0, 0] 
    gmmParameters['meanVectors'][1, :] = [0, 30]
    gmmParameters['meanVectors'][2, :] = [30, 0]   
    gmmParameters['meanVectors'][3, :] = [30, 30]         
    gmmParameters['covMatrices'] = np.zeros((4, 2, 2))
    gmmParameters['covMatrices'][0,:,:] = np.array([[1, -3], [-3, 1]])
    gmmParameters['covMatrices'][1,:,:] = np.array([[8, 4], [4, 8]])
    gmmParameters['covMatrices'][2,:,:] = np.array([[6, 3], [3, 6]])
    gmmParameters['covMatrices'][3,:,:] = np.array([[7, 1], [1, 7]])
    x,labels = generateDataFromGMM(N,gmmParameters)
    return x, labels

def generateDataFromGMM(N,gmmParameters):
#    Generates N vector samples from the specified mixture of Gaussians
#    Returns samples and their component labels
#    Data dimensionality is determined by the size of mu/Sigma parameters
    np.random.seed(0)
    priors = gmmParameters['priors'] # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[1] # Data dimensionality
    C = len(priors) # Number of components
    x = np.zeros((n,N))
    labels = np.zeros((1,N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1,N))
    thresholds = np.zeros((1,C+1))
    thresholds[:,0:C] = np.cumsum(priors)
    thresholds[:,C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:,l]))
        print(indl[0])
        Nl = len(indl[1])
        labels[indl] = (l)*1
        u[indl] = 1.1
        x[:,indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[l,:], covMatrices[l,:,:], Nl))
        
    return x,labels

def plot(data, labels, mark="o"):
    plt.scatter(data[0,labels == 0], data[1,labels == 0], marker=mark, color = "b")
    plt.scatter(data[0,labels == 1], data[1,labels == 1], marker=mark, color = "r")
    plt.scatter(data[0,labels == 2], data[1,labels == 2], marker=mark, color = "g")
    plt.scatter(data[0,labels == 3], data[1,labels == 3], marker=mark, color = "y")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title('Training Dataset')
    plt.show()

create_datasets()