import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import b
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
np.set_printoptions(threshold=np.inf)
plt.rcParams['figure.figsize'] = [9,9]

N_features = 2            # Number of features
N_labels = 2              # Number of classes
N_mixtures = 3          # Number of gaussian distributions

#Seed to obtain same results for random numbers
np.random.seed(10)

# Class Priors
priors = np.array([0.65, 0.35])

# Mean vectors
mean_matrix = np.zeros(shape=[N_mixtures, N_features])
mean_matrix [0, :] = [3, 0]
mean_matrix [1, :] = [0, 3]
mean_matrix [2, :] = [2, 2]

# Covariance matrices
covariance_matrix = np.zeros(shape=[N_mixtures, N_features, N_features])
covariance_matrix [0, :, :] = [[2, 0],[0, 1]]
covariance_matrix [1, :, :] = [[1, 0],[0, 2]]
covariance_matrix [2, :, :] = [[1, 0],[0, 1]]

# Prior weights for gaussian components of class 2 and assigning labels
weights = [0.5,0.5]
cumsum = np.cumsum(priors)

#Function to generate data samples
def generateData(N_Samples):
    label = (np.random.rand(N_Samples) >= priors[1]).astype(int)

    # Generate gaussian distribution for 10000 samples using mean and covariance matrices for each label
    X = np.zeros(shape = [N_Samples, N_features])
    for i in range(N_Samples):
        if (label[i] == 0):
            # Split samples based on mixture weights
            if (np.random.rand(1,1) >= weights[1]):
                X[i, :] = np.random.multivariate_normal(mean_matrix[0, :], covariance_matrix[0, :, :])
            else:
                X[i, :] = np.random.multivariate_normal(mean_matrix[1, :], covariance_matrix[1, :, :])
            
        elif (label[i] == 1):
            X[i, :] = np.random.multivariate_normal(mean_matrix[2, :], covariance_matrix[2, :, :])

    return X, label

# Plot the data distribution to visualize the samples
def plotDistribution(X, label, w, type):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[np.where(label==0),0],X[np.where(label==0),1],color = 'blue')
    ax.scatter(X[np.where(label==1),0],X[np.where(label==1),1],color = 'red')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Data Distribution')
    plot_decision_boundary(X, w, type)
    plt.show()

# Function to plot classified data along with decision boundary
def plot_classified_data(X, label, decision, w, type):
    plt.plot(X[np.where((label == 1) & (decision == 1)),0], X[np.where((label == 1) & (decision == 1)),1], color ='green', marker = '^', markersize = 1)
    plt.plot(X[np.where((label == 1) & (decision == 0)),0], X[np.where((label == 1) & (decision == 0)),1], color ='red', marker = '^', markersize = 1)
    plt.plot(X[np.where((label == 0) & (decision == 1)),0], X[np.where((label == 0) & (decision == 1)),1], color ='red', marker = 's', markersize = 1)
    plt.plot(X[np.where((label == 0) & (decision == 0)),0], X[np.where((label == 0) & (decision == 0)),1], color ='green', marker = 's', markersize = 1)
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.title("Plot showing decision boundary and classified data")
    plot_decision_boundary(X, w, type)

# Function to decide decision boundary based on classifier type
def plot_decision_boundary(X, w, type):
    hgrid = np.linspace(min(X[:,0]), max(X[:,0]), 100)
    vgrid = np.linspace(min(X[:,1]), max(X[:,1]), 100)
    x1, x2 = np.meshgrid(hgrid,vgrid)
    b = np.zeros((100,100))
    loggamma = np.log(priors[0] / priors[1])
    for i in range(len(x1)):
        for j in range(len(x2)):
            if (type == 'T'):
                GaussPDF0 = np.log((w[0] * (multivariate_normal.pdf(np.array([x1[i][j], x2[i][j]]), mean = mean_matrix[0, :], cov = covariance_matrix[0, :,:]))) + (w[1] * (multivariate_normal.pdf(np.array(x1[i][j], x2[i][j]),mean = mean_matrix[1, :], cov = covariance_matrix[1, :,:]))))
                GaussPDF1 = np.log(multivariate_normal.pdf(np.array([x1[i][j], x2[i][j]]), mean = mean_matrix[2, :], cov = covariance_matrix[2, :,:]))
                b[i][j] = GaussPDF1 - GaussPDF0 - loggamma
            elif (type == 'L'):
                z = np.c_[1, x1[i][j], x2[i][j]].T
                b[i][j] = np.sum(np.dot(w.T, z))
            elif (type == 'Q'):
                z = np.c_[1, x1[i][j], x2[i][j], x1[i][j]**2, x1[i][j] * x2[i][j], x2[i][j] ** 2].T
                b[i][j] = np.sum(np.dot(w.T, z))
    
    plt.contour(x1, x2, b, levels = [0])
    plt.show()

# Generate Validation dataset
X_validate, Label_validate = generateData(10000)

## Part A - Theoretical Bayesian Classifier 

# Compute discriminant score using class conditional PDF
GaussPDF0 = np.log(weights[0] * (multivariate_normal.pdf(X_validate,mean = mean_matrix[0, :], cov = covariance_matrix[0, :,:])) + weights[1] * (multivariate_normal.pdf(X_validate,mean = mean_matrix[1, :], cov = covariance_matrix[1, :,:])))
GaussPDF1 = np.log(multivariate_normal.pdf(X_validate,mean = mean_matrix[2, :], cov = covariance_matrix[2, :,:]))
discrim_score = GaussPDF1 - GaussPDF0

# Sort tau values to navigate from minimum to maximum value
sorted_tau = np.sort(discrim_score)
tau_sweep = []

# Calculate mid-points which will be used as threshold values
for i in range(0,9999):
        tau_sweep.append((sorted_tau[i] + sorted_tau[i+1])/2.0)

# Array initialization for results
decision = []
TP = [None] * len(tau_sweep)
FP = [None] * len(tau_sweep)
minPerror = [None] * len(tau_sweep)

# Classify for each threshold and compute error and evaluation metrics
for (index, tau) in enumerate(tau_sweep):
        decision = (discrim_score >= tau)
        TP[index] = (np.size(np.where((decision == 1) & (Label_validate == 1))))/np.size(np.where(Label_validate == 1))
        FP[index] = (np.size(np.where((decision == 1) & (Label_validate == 0))))/np.size(np.where(Label_validate == 0))
        minPerror[index] = (priors[0] * FP[index]) + (priors[1] * (1 - TP[index]))

# Theoretical classification based on class priors
loggamma_ideal = np.log(priors[0] / priors[1])
ideal_decision = (discrim_score >= loggamma_ideal)
TP_ideal = (np.size(np.where((ideal_decision == 1) & (Label_validate == 1))))/np.size(np.where(Label_validate == 1))
FP_ideal = (np.size(np.where((ideal_decision == 1) & (Label_validate == 0))))/np.size(np.where(Label_validate == 0))
minPerror_ideal = (priors[0] * FP_ideal) + (priors[1] * (1 - TP_ideal))
print("Gamma Ideal - %f and corresponding minimum error %f" %(np.exp(loggamma_ideal), minPerror_ideal))

plot_classified_data(X_validate, Label_validate, ideal_decision, weights, 'T')

# Plot ROC curve
plt.plot(FP, TP, color = 'red')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC Curve')
plt.plot(FP[np.argmin(minPerror)], TP[np.argmin(minPerror)],'o',color = 'black')
plt.show()

print("Gamma Practical - %f and corresponding minimum error %f" %(np.exp(tau_sweep[np.argmin(minPerror)]), np.min(minPerror)))

# Loss Function for Logistic Regression
def cost_function(w, z, N, Label_train):
    h = 1 / (1 + np.exp(-(np.dot(w.T, z))))
    cost = (-1 / N) * ((np.sum(Label_train * np.log(h))) + (np.sum((1 - Label_train) * np.log(1 - h))))
    return cost

## Part B- Logistic Based Linear and Quadratic Classifiers
N_train = [20, 200, 2000]                # Number of training samples for each model
for N in N_train:
    # Generate Training Data
    X_train, Label_train = generateData(N)
    # Create basis vector
    z_L = np.column_stack((np.ones(X_train.shape[0]), X_train)).T
    z_Q = np.column_stack((np.ones(X_train.shape[0]), X_train[:, 0], X_train[:, 1], np.square(X_train[:, 0]), np.multiply(X_train[:, 0], X_train[:, 1]), np.square(X_train[:, 1]))).T
    #Initialize weight vector
    w_L = np.zeros(shape = (3, 1))
    w_Q = np.zeros(shape = (6, 1))
    #Perform Optimization using cost function defined
    res_L = minimize(cost_function, w_L, args = (z_L, N, Label_train), method = 'Nelder-Mead')
    res_Q = minimize(cost_function, w_Q, args = (z_Q, N, Label_train), method = 'Nelder-Mead')
    wL_final = res_L.x
    wQ_final = res_Q.x

    # Visualize the training data with decision boundary
    plotDistribution(X_train, Label_train, wL_final, 'L')
    plotDistribution(X_train, Label_train, wQ_final, 'Q')

    # Create Validation dataset
    X_validate, Label_validate = generateData(10000)
    ztest_L = np.column_stack((np.ones(X_validate.shape[0]), X_validate)).T
    # Make decisions to test the models
    decision_L = []
    decision_L = (1 / (1+ np.exp(-(np.dot(wL_final.T, ztest_L)))) >= 0.5).astype(int)

    ztest_Q = np.column_stack((np.ones(X_validate.shape[0]), X_validate[:, 0], X_validate[:, 1], np.square(X_validate[:, 0]), np.multiply(X_validate[:, 0], X_validate[:, 1]), np.square(X_validate[:, 1]))).T
    decision_Q = []
    decision_Q = (1 / (1+ np.exp(-(np.dot(wQ_final.T, ztest_Q)))) >= 0.5).astype(int)

    # Evaluate the model and compute error
    TP_L = (np.size(np.where((decision_L == 1) & (Label_validate == 1))))
    TN_L = (np.size(np.where((decision_L == 0) & (Label_validate == 0))))
    print("Error for Logistic Linear Model with ", N, " samples is ", (10000 - (TP_L + TN_L))/100)

    TP_Q = (np.size(np.where((decision_Q == 1) & (Label_validate == 1))))
    TN_Q = (np.size(np.where((decision_Q == 0) & (Label_validate == 0))))
    print("Error for Logistic Quadratic Model with ", N, " samples is ", (10000 - (TP_Q + TN_Q))/100)

    # Plot Validation dataset with decision boundary
    plot_classified_data(X_validate, Label_validate, decision_L, wL_final, 'L')
    plot_classified_data(X_validate, Label_validate, decision_Q, wQ_final, 'Q')



    
