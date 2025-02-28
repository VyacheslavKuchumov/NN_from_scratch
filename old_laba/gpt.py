import numpy as np

# Define the sigmoid activation function and its derivative.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # x is assumed to be the output of the sigmoid function.
    return x * (1 - x)

# Prepare the dataset for the function NOT((X1 OR X2) AND X3)
# Truth table for inputs (X1, X2, X3) and the corresponding output.
# For clarity, compute the expected output:
#   Let A = (X1 OR X2)
#   Let B = (A AND X3)
#   Then output = NOT(B)
# The truth table:
# X1 X2 X3 | Output
#  0  0  0 |   1
#  0  0  1 |   1
#  0  1  0 |   1
#  0  1  1 |   0
#  1  0  0 |   1
#  1  0  1 |   0
#  1  1  0 |   1
#  1  1  1 |   0

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

y = np.array([[1],
              [1],
              [1],
              [0],
              [1],
              [0],
              [1],
              [0]])

# Set a random seed for reproducibility
np.random.seed(42)

# Define the network architecture
input_dim = 3      # Three input neurons: X1, X2, X3
hidden_dim = 3     # Three neurons in the hidden layer
output_dim = 1     # One output neuron

# Initialize weights with random values between -1 and 1
W1 = 2 * np.random.random((input_dim, hidden_dim)) - 1  # Weights from input to hidden layer
W2 = 2 * np.random.random((hidden_dim, output_dim)) - 1 # Weights from hidden to output layer

# Set learning rate and number of epochs for training
learning_rate = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    # Forward propagation
    layer0 = X
    layer1_input = np.dot(layer0, W1)
    layer1 = sigmoid(layer1_input)
    layer2_input = np.dot(layer1, W2)
    layer2 = sigmoid(layer2_input)
    
    # Compute error at the output layer
    layer2_error = y - layer2
    
    # Optionally, print error every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Mean Absolute Error: {np.mean(np.abs(layer2_error))}")
    
    # Backpropagation
    # Compute delta for output layer
    layer2_delta = layer2_error * sigmoid_derivative(layer2)
    
    # Compute error for hidden layer
    layer1_error = layer2_delta.dot(W2.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)
    
    # Update weights using gradient descent
    W2 += learning_rate * layer1.T.dot(layer2_delta)
    W1 += learning_rate * layer0.T.dot(layer1_delta)

# After training, print the final outputs of the network for the dataset
print("\nFinal outputs after training:")
print(layer2)

# Optionally, compare the network outputs with the expected outputs
print("\nExpected outputs:")
print(y)
