import numpy as np
import pygame
import sys

# -----------------------
# Neural Network Training
# -----------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # x is assumed to be the output of the sigmoid function.
    return x * (1 - x)

# Prepare the dataset for NOT((X1 OR X2) AND X3)
# Truth table:
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

# Define network architecture
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
    # Forward propagation on entire dataset
    layer0 = X
    layer1_input = np.dot(layer0, W1)
    layer1 = sigmoid(layer1_input)
    layer2_input = np.dot(layer1, W2)
    layer2 = sigmoid(layer2_input)
    
    # Compute error at the output layer
    layer2_error = y - layer2
    
    # Optionally print error every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Mean Absolute Error: {np.mean(np.abs(layer2_error))}")
    
    # Backpropagation
    layer2_delta = layer2_error * sigmoid_derivative(layer2)
    layer1_error = layer2_delta.dot(W2.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)
    
    # Update weights with gradient descent
    W2 += learning_rate * layer1.T.dot(layer2_delta)
    W1 += learning_rate * layer0.T.dot(layer1_delta)

print("\nFinal outputs after training:")
print(layer2)
print("\nExpected outputs:")
print(y)

# -----------------------
# Pygame Visualization
# -----------------------

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network Interactive Visualization")
font = pygame.font.SysFont("Arial", 20)
clock = pygame.time.Clock()

# Define positions for nodes in each layer:
# Input layer positions (left side)
input_positions = [(150, 200), (150, 300), (150, 400)]
# Hidden layer positions (center)
hidden_positions = [(400, 150), (400, 300), (400, 450)]
# Output layer position (right side)
output_positions = [(650, 300)]
node_radius = 40

# Initial input values (can be toggled by user)
input_values = [0, 0, 0]

def forward_propagation(inputs):
    """Perform forward propagation for a given input array (1x3)."""
    l0 = np.array([inputs])
    l1 = sigmoid(np.dot(l0, W1))
    l2 = sigmoid(np.dot(l1, W2))
    return l1[0], l2[0]

def draw_node(pos, activation, label_text):
    """
    Draw a circle at pos.
    Activation is used to determine color intensity (for hidden and output nodes).
    label_text is the text to display inside the circle.
    """
    # Map activation (0 to 1) to a color intensity (for visualization)
    # For input nodes, activation is either 0 or 1, and we use green for 1.
    if label_text in ["X1", "X2", "X3"]:
        color = (0, 255, 0) if activation >= 1 else (255, 255, 255)
    else:
        # For hidden/output nodes, blue intensity proportional to activation
        blue_intensity = int(activation * 255)
        color = (100, 100, blue_intensity)
    
    pygame.draw.circle(screen, color, pos, node_radius)
    pygame.draw.circle(screen, (0, 0, 0), pos, node_radius, 2)  # outline
    
    text_surface = font.render(label_text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(center=pos)
    screen.blit(text_surface, text_rect)

def draw_connection(start_pos, end_pos, weight):
    """
    Draw a line representing a connection between two nodes.
    The weight is displayed near the line.
    """
    pygame.draw.line(screen, (50, 50, 50), start_pos, end_pos, 2)
    # Position the weight text at the midpoint of the connection
    mid_point = ((start_pos[0] + end_pos[0]) // 2, (start_pos[1] + end_pos[1]) // 2)
    weight_text = f"{weight:.2f}"
    text_surface = font.render(weight_text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(center=mid_point)
    screen.blit(text_surface, text_rect)

def draw_network(input_vals, hidden_acts, output_act):
    """Draw the complete network with the given activations."""
    screen.fill((220, 220, 220))
    
    # Draw connections from input to hidden layer
    for i, ipos in enumerate(input_positions):
        for j, hpos in enumerate(hidden_positions):
            draw_connection((ipos[0] + node_radius, ipos[1]),
                            (hpos[0] - node_radius, hpos[1]),
                            W1[i, j])
    
    # Draw connections from hidden to output layer
    for j, hpos in enumerate(hidden_positions):
        for k, opos in enumerate(output_positions):
            draw_connection((hpos[0] + node_radius, hpos[1]),
                            (opos[0] - node_radius, opos[1]),
                            W2[j, k])
    
    # Draw input nodes (labels X1, X2, X3)
    for i, pos in enumerate(input_positions):
        draw_node(pos, input_vals[i], f"X{i+1}:{input_vals[i]}")
    
    # Draw hidden nodes (labels H1, H2, H3 with activation values)
    for j, pos in enumerate(hidden_positions):
        draw_node(pos, hidden_acts[j], f"H{j+1}:{hidden_acts[j]:.2f}")
    
    # Draw output node (label O1 with activation value)
    for k, pos in enumerate(output_positions):
        draw_node(pos, output_act[k], f"O{1}:{output_act[k]:.2f}")
    
    # Instructions
    instructions = [
        "Press 1, 2, 3 keys to toggle input values X1, X2, X3.",
        "Press ESC or close window to exit."
    ]
    for idx, line in enumerate(instructions):
        text_surface = font.render(line, True, (0, 0, 0))
        screen.blit(text_surface, (50, 20 + idx * 25))
    
    pygame.display.flip()

# Main loop for interactive visualization
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                break
            # Toggle inputs: keys 1, 2, 3
            if event.key == pygame.K_1:
                input_values[0] = 0 if input_values[0] == 1 else 1
            if event.key == pygame.K_2:
                input_values[1] = 0 if input_values[1] == 1 else 1
            if event.key == pygame.K_3:
                input_values[2] = 0 if input_values[2] == 1 else 1
    
    # Compute activations using the current input values
    hidden_activations, output_activation = forward_propagation(input_values)
    
    # Draw the network with the updated activations
    draw_network(input_values, hidden_activations, output_activation)
    
    clock.tick(10)  # Limit to 10 frames per second

pygame.quit()
sys.exit()
