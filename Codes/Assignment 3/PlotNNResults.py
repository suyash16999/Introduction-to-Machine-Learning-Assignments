import numpy as np
import matplotlib.pyplot as plt

# x represents number of training samples
x = [100, 200, 500, 1000, 2000, 5000]
# elements in error correspond to loss produced by ML model with training samples in x
error = [23.65, 20.075, 19.75, 17.93, 17.96, 17.91]

# Plot graph to compare theoretical classifier and various MLP models
plt.semilogx(x, error, marker = ".", markersize = 15)
plt.axhline(y = 16.55, color = 'r', linestyle = '-')
plt.xlabel("Semilog Axis representing number of samples")
plt.ylabel("Emperically Estimated P Error")
plt.title("Plot to Compare Errors of the trained models")
plt.show()
