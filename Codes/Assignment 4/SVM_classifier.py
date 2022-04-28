import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Importing training dataset
dataset_train = pd.read_csv('SVM_Dtrain_.csv')

# Extract features in an array
Xtrain = dataset_train.iloc[:, 0:2].values
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)

# Extract training labels
ytrain = dataset_train.iloc[:, 2].values

# Importing test dataset
dataset_test = pd.read_csv('SVM_Dtest_.csv')

# Split features and labels
Xtest = dataset_test.iloc[:, 0:2].values
Xtest = sc.transform(Xtest)
ytest = dataset_test.iloc[:, 2].values

# Define range of hyperparameters for cross validation
C = [0.01, 0.1, 1, 10, 100, 1000, 10000]
gamma = [0.1, 0.01, 0.001, 0.0001]
hyper_parameters = {"kernel": ["rbf"], "gamma": gamma, "C": C}

# Perform cross validation and identify best results 
clf = GridSearchCV(SVC(), hyper_parameters, cv = KFold(n_splits=10))
clf.fit(Xtrain, ytrain)
print(clf.best_params_)

# Print average accuracy scores for all combinations
means = clf.cv_results_["mean_test_score"]
stds = clf.cv_results_["std_test_score"]
for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

scores = np.array(means).reshape(7, 4)

plt.figure(figsize=(10, 10))
plt.subplots_adjust(left = 0.15, right = 0.6, bottom = 0.3, top = 0.95)
plt.imshow(scores, interpolation='nearest', cmap = plt.cm.nipy_spectral)
plt.xlabel('gamma')
plt.ylabel('C')
plt.title("Color Map for selecting hyperparameters")
plt.xticks(np.arange(len(gamma)), gamma, rotation=45)
plt.yticks(np.arange(len(C)), C)
plt.colorbar()
plt.show()

# Train final SVM classifier
classifier = SVC(C=10, kernel = 'rbf', gamma = 0.1, random_state = 0)
classifier.fit(Xtrain, ytrain)

# Test the model with appropriate evaulation metrics
ypred = classifier.predict(Xtest)
print(ypred)
cm = confusion_matrix(ytest, ypred)
print(cm)
print("The accuracy of the model is  {} %".format(str(round(accuracy_score(ytest,ypred),4)*100)))

# Define range of points for identifying classification boundary
x1_min, x1_max = Xtest[:, 0].min() - 1, Xtest[:, 0].max() + 1
x2_min, x2_max = Xtest[:, 1].min() - 1, Xtest[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max), np.arange(x2_min, x2_max))
print(x1_max, x1_min)
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot classification boundary superimposed over testing data
plt.scatter(Xtest[:, 0], Xtest[:, 1], c = ytest, cmap = "brg", s = 20, edgecolors = 'y')
plt.contourf(xx, yy, Z, cmap = "brg", alpha = 0.3)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Plot showing decision boundary")
plt.show()
