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

# Define network architecture with two hidden layers
input_dim = 3        # Three input neurons: X1, X2, X3
hidden1_dim = 3      # First hidden layer: 3 neurons
hidden2_dim = 3      # Second hidden layer: 3 neurons
output_dim = 1       # One output neuron

# Initialize weights with random values between -1 and 1
W1 = 2 * np.random.random((input_dim, hidden1_dim)) - 1      # Input -> Hidden Layer 1
W2 = 2 * np.random.random((hidden1_dim, hidden2_dim)) - 1      # Hidden Layer 1 -> Hidden Layer 2
W3 = 2 * np.random.random((hidden2_dim, output_dim)) - 1       # Hidden Layer 2 -> Output

learning_rate = 0.1
epochs = 10000

# Training loop with two hidden layers
for epoch in range(epochs):
    # Forward propagation
    layer0 = X
    layer1_input = np.dot(layer0, W1)
    layer1 = sigmoid(layer1_input)             # Activation for Hidden Layer 1
    
    layer2_input = np.dot(layer1, W2)
    layer2 = sigmoid(layer2_input)             # Activation for Hidden Layer 2
    
    layer3_input = np.dot(layer2, W3)
    layer3 = sigmoid(layer3_input)             # Output layer activation
    
    # Compute error at the output layer
    layer3_error = y - layer3
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Mean Absolute Error: {np.mean(np.abs(layer3_error))}")
    
    # Backpropagation
    layer3_delta = layer3_error * sigmoid_derivative(layer3)
    
    layer2_error = layer3_delta.dot(W3.T)
    layer2_delta = layer2_error * sigmoid_derivative(layer2)
    
    layer1_error = layer2_delta.dot(W2.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)
    
    # Update weights using gradient descent
    W3 += learning_rate * layer2.T.dot(layer3_delta)
    W2 += learning_rate * layer1.T.dot(layer2_delta)
    W1 += learning_rate * layer0.T.dot(layer1_delta)

print("\nFinal outputs after training:")
print(layer3)
print("\nExpected outputs:")
print(y)

# -----------------------
# Pygame Visualization Setup
# -----------------------

pygame.init()
WIDTH, HEIGHT = 1200, 600  # Increase width to accommodate truth table on the right
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network Interactive Visualization with Truth Table")
font = pygame.font.SysFont("Arial", 20)
clock = pygame.time.Clock()

# Define positions for nodes in each layer:
# Input layer positions (left side)
input_positions = [(150, 200), (150, 300), (150, 400)]
# Hidden Layer 1 positions (next column)
hidden1_positions = [(350, 150), (350, 300), (350, 450)]
# Hidden Layer 2 positions (next column)
hidden2_positions = [(550, 150), (550, 300), (550, 450)]
# Output layer position (right side)
output_positions = [(750, 300)]
node_radius = 40

# Initial input values (can be toggled by user)
input_values = [0, 0, 0]

# Truth table for display: each row is ([X1, X2, X3], target)
truth_table = [
    ([0, 0, 0], 1),
    ([0, 0, 1], 1),
    ([0, 1, 0], 1),
    ([0, 1, 1], 0),
    ([1, 0, 0], 1),
    ([1, 0, 1], 0),
    ([1, 1, 0], 1),
    ([1, 1, 1], 0)
]

def forward_propagation(inputs):
    """Perform forward propagation for a given input array (1x3).
       Returns activations for Hidden Layer 1, Hidden Layer 2, and Output."""
    l0 = np.array([inputs])
    l1 = sigmoid(np.dot(l0, W1))
    l2 = sigmoid(np.dot(l1, W2))
    l3 = sigmoid(np.dot(l2, W3))
    return l1[0], l2[0], l3[0]

def draw_node(pos, activation, label_text):
    """
    Draw a circle at pos.
    For input nodes, use green if active; for hidden and output nodes, use blue intensity based on activation.
    """
    if label_text.startswith("X"):
        color = (0, 255, 0) if activation >= 1 else (255, 255, 255)
    else:
        blue_intensity = int(activation * 255)
        color = (100, 100, blue_intensity)
    
    pygame.draw.circle(screen, color, pos, node_radius)
    pygame.draw.circle(screen, (0, 0, 0), pos, node_radius, 2)
    
    text_surface = font.render(label_text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(center=pos)
    screen.blit(text_surface, text_rect)

def draw_connection(start_pos, end_pos, weight):
    """
    Draw a line representing a connection between two nodes.
    The weight is displayed near the line.
    """
    pygame.draw.line(screen, (50, 50, 50), start_pos, end_pos, 2)
    mid_point = ((start_pos[0] + end_pos[0]) // 2, (start_pos[1] + end_pos[1]) // 2)
    weight_text = f"{weight:.2f}"
    text_surface = font.render(weight_text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(center=mid_point)
    screen.blit(text_surface, text_rect)

def draw_network(input_vals, hidden1_acts, hidden2_acts, output_act):
    """Draw the complete network with activations from all layers."""
    # Fill background
    screen.fill((220, 220, 220))
    
    # Draw connections: Input -> Hidden Layer 1
    for i, ipos in enumerate(input_positions):
        for j, hpos in enumerate(hidden1_positions):
            draw_connection((ipos[0] + node_radius, ipos[1]),
                            (hpos[0] - node_radius, hpos[1]),
                            W1[i, j])
    
    # Draw connections: Hidden Layer 1 -> Hidden Layer 2
    for i, h1pos in enumerate(hidden1_positions):
        for j, h2pos in enumerate(hidden2_positions):
            draw_connection((h1pos[0] + node_radius, h1pos[1]),
                            (h2pos[0] - node_radius, h2pos[1]),
                            W2[i, j])
    
    # Draw connections: Hidden Layer 2 -> Output
    for i, h2pos in enumerate(hidden2_positions):
        for j, opos in enumerate(output_positions):
            draw_connection((h2pos[0] + node_radius, h2pos[1]),
                            (opos[0] - node_radius, opos[1]),
                            W3[i, j])
    
    # Draw nodes for each layer:
    # Input layer nodes
    for i, pos in enumerate(input_positions):
        draw_node(pos, input_vals[i], f"X{i+1}:{input_vals[i]}")
    
    # Hidden Layer 1 nodes
    for i, pos in enumerate(hidden1_positions):
        draw_node(pos, hidden1_acts[i], f"H1_{i+1}:{hidden1_acts[i]:.2f}")
    
    # Hidden Layer 2 nodes
    for i, pos in enumerate(hidden2_positions):
        draw_node(pos, hidden2_acts[i], f"H2_{i+1}:{hidden2_acts[i]:.2f}")
    
    # Output node(s)
    for i, pos in enumerate(output_positions):
        draw_node(pos, output_act[i], f"O1:{output_act[i]:.2f}")

def draw_truth_table():
    """Draw the truth table on the right side of the screen."""
    table_x = 950  # Starting x coordinate for truth table
    table_y = 50   # Starting y coordinate
    header = ["X1", "X2", "X3", "Target"]
    # Draw header
    for i, text in enumerate(header):
        text_surface = font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(table_x, table_y + i * 25))
        screen.blit(text_surface, text_rect)
    
    # Draw horizontal line under header
    pygame.draw.line(screen, (0, 0, 0), (table_x - 40, table_y + 30), (table_x + 40, table_y + 30), 2)
    
    # Draw each row of the truth table
    for row_index, (inputs, target) in enumerate(truth_table):
        row_y = table_y + 40 + row_index * 25
        row_values = inputs + [target]
        row_text = "   ".join(str(val) for val in row_values)
        text_surface = font.render(row_text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(table_x, row_y))
        screen.blit(text_surface, text_rect)

def draw_instructions():
    """Draw instructions at the top of the screen."""
    instructions = [
        "Press 1, 2, 3 keys to toggle input values X1, X2, X3.",
        "Press ESC or close window to exit."
    ]
    for idx, line in enumerate(instructions):
        text_surface = font.render(line, True, (0, 0, 0))
        screen.blit(text_surface, (50, 20 + idx * 25))

# Main loop for interactive visualization
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                break
            # Toggle inputs with keys 1, 2, 3
            if event.key == pygame.K_1:
                input_values[0] = 0 if input_values[0] == 1 else 1
            if event.key == pygame.K_2:
                input_values[1] = 0 if input_values[1] == 1 else 1
            if event.key == pygame.K_3:
                input_values[2] = 0 if input_values[2] == 1 else 1
    
    # Compute activations for current input values
    hidden1_acts, hidden2_acts, output_act = forward_propagation(input_values)
    
    # Draw the network on the left side
    draw_network(input_values, hidden1_acts, hidden2_acts, output_act)
    
    # Draw instructions and truth table
    draw_instructions()
    draw_truth_table()
    
    pygame.display.flip()
    clock.tick(10)  # Limit frame rate to 10 FPS

pygame.quit()
sys.exit()
