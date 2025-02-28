import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# The truth table:
X = np.array([[1, 0, 0, 0],
              [1, 0, 0, 1],
              [1, 0, 1, 0],
              [1, 0, 1, 1],
              [1, 1, 0, 0],
              [1, 1, 0, 1],
              [1, 1, 1, 0],
              [1, 1, 1, 1]])

# The expected output:
y = np.array([[1],
              [1],
              [1],
              [0],
              [1],
              [0],
              [1],
              [0]])

input_dim = 4      
hidden_dim = 3     
output_dim = 1    

W1 = 2 * np.random.random((input_dim, hidden_dim)) - 1
W2 = 2 * np.random.random((hidden_dim, output_dim)) - 1

learning_rate = 0.8
epochs = 2000

errors = []

for epoch in range(epochs):
    # Forward propagation
    layer0 = X
    layer1 = sigmoid(np.dot(layer0, W1))
    layer2 = sigmoid(np.dot(layer1, W2))

    # Compute error
    layer2_error = y - layer2
    mean_error = np.mean(np.abs(layer2_error))  # Mean Absolute Error (MAE)
    errors.append(mean_error)

    # Backpropagation
    layer2_delta = layer2_error * sigmoid_derivative(layer2)
    layer1_error = layer2_delta.dot(W2.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    # Update weights
    W2 += learning_rate * layer1.T.dot(layer2_delta)
    W1 += learning_rate * layer0.T.dot(layer1_delta)

# Plot error over epochs
plt.plot(range(epochs), errors, label="Error")
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.title("Training Error over Epochs")
plt.legend()
plt.show()
