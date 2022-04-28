import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_datasets():
    Ntrain = [100,200,500,1000,2000,5000]
    for N in Ntrain:
        data, labels = generateData(N)
        labels = np.squeeze(labels)
        plot3(data, labels)
        dataset = pd.DataFrame(np.transpose(np.vstack((data, labels))))
        filename = 'Dtrain_' + str(N) +'.csv'
        dataset.to_csv(filename, index = False)
    Ntest = 100000
    data, labels = generateData(Ntest)
    labels = np.squeeze(labels)
    plot3(data, labels)
    dataset = pd.DataFrame(np.transpose(np.vstack((data, labels))))
    filename = 'Dtest_' + str(Ntest) +'.csv'
    dataset.to_csv(filename, index = False)


def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.25,.25,.25,.25] # priors should be a row vector
    gmmParameters['meanVectors'] = np.zeros((4,3))
    gmmParameters['meanVectors'][0, :] = [0, 0, 0] 
    gmmParameters['meanVectors'][1, :] = [0, 0, 3]
    gmmParameters['meanVectors'][2, :] = [0, 3, 0]   
    gmmParameters['meanVectors'][3, :] = [3, 0, 0]         
    gmmParameters['covMatrices'] = np.zeros((4, 3, 3))
    gmmParameters['covMatrices'][0,:,:] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][1,:,:] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][2,:,:] = np.array([[1, 0, -3], [0, .5, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][3,:,:] = np.array([[8, 0, 0], [0, 1, 0], [3, 0, 15]])
    print(gmmParameters['covMatrices'])
    x,labels = generateDataFromGMM(N,gmmParameters)
    return x, labels

def generateDataFromGMM(N,gmmParameters):
#    Generates N vector samples from the specified mixture of Gaussians
#    Returns samples and their component labels
#    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors'] # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[1] # Data dimensionality
    C = len(priors) # Number of components
    print(C)
    x = np.zeros((n,N))
    labels = np.zeros((1,N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1,N))
    thresholds = np.zeros((1,C+1))
    thresholds[:,0:C] = np.cumsum(priors)
    thresholds[:,C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:,l]))
        Nl = len(indl[1])
        labels[indl] = (l)*1
        u[indl] = 1.1
        x[:,indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[l,:], covMatrices[l,:,:], Nl))
        
    return x,labels

def plot3(data, labels, mark="o"):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[0,labels == 0], data[1,labels == 0], data[2,labels == 0], marker=mark, color = "b")
    ax.scatter(data[0,labels == 1], data[1,labels == 1], data[2,labels == 1], marker=mark, color = "r")
    ax.scatter(data[0,labels == 2], data[1,labels == 2], data[2,labels == 2], marker=mark, color = "g")
    ax.scatter(data[0,labels == 3], data[1,labels == 3], data[2,labels == 3], marker=mark, color = "y")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title('Training Dataset')
    plt.show()

create_datasets()