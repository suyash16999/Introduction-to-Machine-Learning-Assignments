import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import torch
import torch
from torch import LongTensor, nn, sigmoid

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define the layers and activation of Neural Network
class NeuralNetwork(nn.Module):
    # Initialize number of perceptrons in hidden layer as a parameter
    def __init__(self, nodes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, nodes)
        self.fc2 = nn.Linear(nodes, 1)

    # Define forward propogation
    def forward(self, x):
        x = torch.pow(self.fc1(x), 2)
        #x = nn.ELU()(self.fc1(x))
        logits = nn.Sigmoid()(self.fc2(x))
        return logits

# Randomly initialize weights
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

# Import dataset
dataset = pd.read_csv('SVM_Dtrain_.csv')
# Create tensors for features and labels
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values
y = np.array([0.0 if i == -1 else 1.0 for i in y])

X = torch.from_numpy(X)
y = torch.from_numpy(y)

# Define 10 folds for performing cross validation
kf = KFold(n_splits = 10)
test_loss = []

# Vary neurons from 1 to 14 for identifying optimal number in the hidden layer
for neuron in range(1, 15):
    loss_folds = 0
    # Perform cross validation taking each fold as validation set and remaining as training set
    for train_index, test_index in kf.split(X,y):
        Xtrain, ytrain = X[train_index], y[train_index]
        Xtest, ytest = X[test_index], y[test_index]

        # Intialize model with random weights and number of perceptrons in hidden layer
        model = NeuralNetwork(neuron).to(device)
        model.apply(init_weights)

        # Compute minimum binary cross entropy loss
        loss_fn = nn.BCELoss()
        # Optimize the learning parameters using Stochastic Gradient Descent
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum = 0.9)

        # Train each model for multiple epochs
        for epoch in range(1000):
                # Compute training loss
                pred = model(Xtrain.float())
                pred = torch.flatten(pred)
                loss = loss_fn(pred, ytrain.float())

                # Perform Backpropagation to update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ypred = torch.zeros(0,dtype=torch.long, device='cpu')

        print("Final Training Loss:", loss.item())

        # Validate the trained model
        with torch.no_grad():
            pred = model(Xtest.float())
            pred = torch.flatten(pred)
            loss_folds += loss_fn(pred, ytest.float()).item()

    # Average the lossvalidated on each fold after cross validation
    test_loss.append(loss_folds/10)

print(test_loss)
# Plot a bar graph to decide optimal number of perceptrons
plt.bar(np.linspace(1,14,14), test_loss)
plt.xlabel("Number of Perceptrons")
plt.ylabel("Average Loss after 10-Fold cross validation")
plt.title("Plot of Error vs Perceptrons in Hidden Layer")
plt.show()

# Import final test dataset for performance evaluation
test_dataset = pd.read_csv('SVM_Dtest_.csv')
Xtest = test_dataset.iloc[:, 0:2].values
ytest = test_dataset.iloc[:, 2].values
ytest = np.array([0.0 if i == -1 else 1.0 for i in ytest])
Xtest = torch.from_numpy(Xtest)
ytest = torch.from_numpy(ytest)

# Train final model with optimal number of neurons identifed
final_model = NeuralNetwork(6).to(device)
final_model.apply(init_weights)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(final_model.parameters(), lr=1e-3, momentum = 0.9)

for epoch in range(10000):
            # Compute prediction error
            pred = final_model(X.float())
            pred = torch.flatten(pred)
            loss = loss_fn(pred, y.float())

            # Peform backpropogation to update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()

print(f"Final Training loss: {loss:>7f}")
ypred = torch.zeros(0,dtype=torch.long, device='cpu')

# Test the final model and compute evaluation metrics
with torch.no_grad():
    pred = final_model(Xtest.float())
    pred = torch.flatten(pred)
    print(pred)
    ypred = torch.Tensor([0 if value <= 0.5 else 1 for value in pred.detach().numpy()])
    loss = loss_fn(pred, ytest.float()).item()
    accuracy = (ypred == ytest).type(torch.float).sum().item()
    print(loss, accuracy)

    ConfusionMatrix = confusion_matrix(ytest, ypred)
    print(ConfusionMatrix)

    error = (1 - (ConfusionMatrix[0][0] + ConfusionMatrix[1][1]) / 10000)
    print(error)

    x1_min, x1_max = Xtest[:, 0].min() - 1, Xtest[:, 0].max() + 1
    x2_min, x2_max = Xtest[:, 1].min() - 1, Xtest[:, 1].max() + 1
    print(x1_max, x1_min)
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max), np.arange(x2_min, x2_max))

    Z = final_model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())
    Z = np.array([0 if value[0] <= 0.5 else 1 for value in Z.detach().numpy()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c = ytest)
    plt.contour(xx, yy, Z)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Plot showing decision boundary")
    plt.show()
