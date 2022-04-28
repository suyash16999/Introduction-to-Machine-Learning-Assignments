import numpy as np
import sys
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import multivariate_normal
np.set_printoptions(threshold=sys.maxsize)

# Read the image from the specified path
img = cv.imread('GMM_img.jpg')
print(img.shape)

# Downsample the image with specified scale
scale_percent = 60 # percent of original size
w = int(img.shape[1] * (scale_percent / 100))
h = int(img.shape[0] * (scale_percent / 100))
dim = (w, h)
  
# Resize image
img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

# Intialize parameters of the image
N_rows = img.shape[0]
N_columns = img.shape[1]
N_channels = img.shape[2]
N_pixels = N_rows * N_columns
N_features = 5

# Create raw feature vector
feature_vector = np.zeros(shape = (N_pixels, N_features))
pixel = 0 
for row in range(N_rows):
    for col in range(N_columns):
        feature_vector[pixel, 0] = row
        feature_vector[pixel, 1] = col
        for channel in range(N_channels):
            feature_vector[pixel, channel + 2] = img[row, col, channel]
        pixel += 1

# Normalize the feature values
sc = MinMaxScaler()
feature_vector = sc.fit_transform(feature_vector)

# Define number of folds and list to store model orders
kf = KFold(n_splits = 10)
Order = []

Max_likelihood = []
# Test model orders rangine from 1 to 6
for order in range(1,7):
    score = 0
    # Perform 10-fold cross validation and compute log-likelihood in each case
    for train_index, test_index in kf.split(feature_vector):
        Xtrain = feature_vector[train_index]
        Xtest = feature_vector[test_index]
        model = GaussianMixture(n_components = order, init_params = 'random', max_iter = 3000, tol = 1e-8, n_init = 3)
        model.fit(Xtrain)
        score += model.score(Xtest)
        print(score)

    # Compute average score after cross validation
    avg_score = score / 10
    # Store average scores in a list
    Max_likelihood.append(avg_score)

# Plot Log Likelihood Score with respect to number of components in each model
plt.plot(np.linspace(1, 6, 6), Max_likelihood)
plt.xlabel("Number of Components")
plt.ylabel("Average Log Likelihood on Validation Datasets")
plt.title("Plot to Identify Number of Components yielding Maximum Score")
plt.show()

# Final Model Fitting
N_components = 6
model = GaussianMixture(n_components = N_components, init_params='random', max_iter = 4000, tol = 1e-8, n_init = 3)
model.fit(feature_vector)

# Compute class posteriors using model weights and conditional probabilties
posteriors = np.zeros((N_components, N_pixels))
for i in range(N_components):
    PDF = multivariate_normal.pdf(feature_vector, mean = model.means_[i,:], cov = model.covariances_[i,:,:])
    posteriors[i, :] = (model.weights_[i] * PDF)

# Decide label for each pixel with maximum posterior value
img_labels = np.argmax(posteriors, axis = 0)

# Plot segmented image
plt.imshow(img_labels.reshape(N_rows, N_columns))
plt.show()






