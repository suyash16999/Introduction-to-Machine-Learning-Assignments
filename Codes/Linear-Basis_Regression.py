import numpy as np
from numpy.linalg import inv
import numpy.matlib
from matplotlib import pyplot as plt

Ntrain = 100
NValidate = 1000

# Create training and validation dataset
def hw2q2():
    data = generateData(Ntrain)
    plot3(data[0,:],data[1,:],data[2,:], type = 't')
    xTrain = data[0:2,:]
    yTrain = data[2,:]
    
    data = generateData(NValidate)
    plot3(data[0,:],data[1,:],data[2,:], type = 'v')
    xValidate = data[0:2,:]
    yValidate = data[2,:]
    
    return xTrain,yTrain,xValidate,yValidate

# Generate data samples following gaussian distribution
def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3,.4,.3] # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:,:,0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:,:,1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:,:,2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    x,labels = generateDataFromGMM(N,gmmParameters)
    return x

def generateDataFromGMM(N,gmmParameters):
#    Generates N vector samples from the specified mixture of Gaussians
#    Returns samples and their component labels
#    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors'] # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0] # Data dimensionality
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
        Nl = len(indl[1])
        labels[indl] = (l+1)*1
        u[indl] = 1.1
        x[:,indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:,l], covMatrices[:,:,l], Nl))
        
    return x,labels

# Plot a 3D graph showing relationship between input features and target variable
def plot3(a,b,c,type,mark="o",col="b"):
  from matplotlib import pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter(a, b, c,marker=mark,color=col)
  ax.set_xlabel("x1")
  ax.set_ylabel("x2")
  ax.set_zlabel("y")
  if type == 't':
    ax.set_title('Training Dataset')
  elif type == 'v':
        ax.set_title('Validation Dataset')
  plt.show()

# Compute the target variable for the model using input and weight vector
def compute_y(x, w):
    y = np.matmul(w[:,0], pow(x,3)) + np.matmul(w[:,1], pow(x,2)) + np.matmul(w[:,2], pow(x,1)) + np.matmul(w[:,3], np.ones((2,1)))
    return y

# Generate data samples
xTrain,yTrain,xValidate,yValidate = hw2q2()
# Initialize hyperparameters
gamma_array = np.linspace(0.0001, 10000, num = 1000)
sigma_sq = 0.001
noise = np.random.normal(0, sigma_sq)

## Part A - Implementation of ML Estimation
z_train = np.row_stack((pow(xTrain, 3), pow(xTrain, 2), xTrain, np.ones(shape = (xTrain.shape[0],xTrain.shape[1]))))
z = z_train.T.reshape(100,4,2)
A = np.zeros(shape = (2,2))
B = np.zeros(shape = (2,4))

# Model Training
for sample in range(Ntrain):
    A += np.matmul(z[sample,:,:].T, z[sample,:,:])
    B += z[sample,:,:].T * yTrain[sample]

w_ML = np.matmul(inv(A), B)

# Model Evaluation using Mean Sqaured Error
SE = 0
y = np.zeros(shape = (NValidate))
for i in range(NValidate):
    y[i] = compute_y(xValidate[:,i], w_ML)
    SE += pow((yValidate[i] - y[i]), 2)

MSE = SE/1000
print("Maximum Value of MSE-", MSE)

# Implementation of MAP estimation for a range of gamma values
MSE_MAP = np.zeros(shape = (len(gamma_array)))
for idx, gamma in enumerate(gamma_array):
    for sample in range(Ntrain):
        A += (np.matmul(z[sample,:,:].T, z[sample,:,:]) + (sigma_sq / gamma) * np.eye(2))
        B += z[sample,:,:].T * yTrain[sample]

    # Model Evaluation using Mean Sqaured Error
    w_ML = np.matmul(inv(A), B)
    SE = 0
    y = np.zeros(shape = (NValidate))
    for i in range(NValidate):
        y[i] = compute_y(xValidate[:,i], w_ML)
        SE += pow((yValidate[i] - y[i]), 2)

    MSE_MAP[idx] = SE/1000

print("Minimum Value of MSE-", np.min(MSE_MAP), " for corresponding gamma = ", gamma_array[np.argmin(MSE_MAP)])
print("Maximum Value of MSE-", np.max(MSE_MAP), " for corresponding gamma = ", gamma_array[np.argmax(MSE_MAP)])

# Visualize variation in Mean Squared Error with respect to gamma
plt.plot(gamma_array, MSE_MAP)
plt.xlabel("Gamma")
plt.ylabel("Mean Squared Error")
plt.title("Variation in MSE with respect to Gamma")
plt.show()




