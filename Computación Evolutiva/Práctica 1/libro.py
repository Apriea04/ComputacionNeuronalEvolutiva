import numpy as np


# Define the function
def f(x):
    if x < -2:
        return -2.186 * x - 12.864
    elif x >= -2 and x < 0:
        return 4.246 * x
    else:
        return 19 * np.exp(-0.05 * x - 0.5) * np.sin(0.03 * x**2 + 0.7 * x)


# Vectorize the function for array input
f_np = np.vectorize(f)

# Generate data
train_percent = 0.8
cantidad_puntos = 200
x_data = np.linspace(-10, 10, cantidad_puntos)  # For equidistant points
y_data = f_np(x_data)
data = np.array([x_data, y_data]).T
np.random.shuffle(data)

# Create training and test datasets
train_data = data[: int(train_percent * cantidad_puntos)]
test_data = data[int((1 - train_percent) * cantidad_puntos) :]

x_train = train_data[:, 0].reshape(-1, 1)
y_train = train_data[:, 1].reshape(-1, 1)
x_test = test_data[:, 0].reshape(-1, 1)
y_test = test_data[:, 1].reshape(-1, 1)

# Initialize neural network
input_size = 1
hidden_neurons = 50
output_size = 1
epochs = 8000000
eta = 0.00001


# Define activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Initialize weights and biases
weights_input_hidden = np.random.randn(input_size, hidden_neurons)
bias_hidden = np.random.randn(hidden_neurons)
weights_hidden_output = np.random.randn(hidden_neurons, output_size)
bias_output = np.random.randn(output_size)

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    hidden_inputs = np.dot(x_train, weights_input_hidden) + bias_hidden
    hidden_outputs = sigmoid(hidden_inputs)
    output = np.dot(hidden_outputs, weights_hidden_output) + bias_output

    # Backpropagation
    output_error = output - y_train
    output_delta = output_error * sigmoid_derivative(output)

    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_inputs)

    # Update weights and biases
    weights_hidden_output -= eta * np.dot(hidden_outputs.T, output_delta)
    bias_output -= eta * np.sum(output_delta, axis=0)
    weights_input_hidden -= eta * np.dot(x_train.T, hidden_delta)
    bias_hidden -= eta * np.sum(hidden_delta, axis=0)

# Evaluate the neural network
hidden_layer_output = sigmoid(np.dot(x_test, weights_input_hidden) + bias_hidden)
predicted_output = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
mse = np.mean((predicted_output - y_test) ** 2)
print("Mean Squared Error:", mse)

import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
y = f_np(x)
# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x, y, color="blue", label="Actual")
plt.scatter(x_test, predicted_output, color="red", label="Predicted")
plt.title("Actual vs. Predicted")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.legend()
plt.grid(True)
plt.show()
