# client.py
import flwr as fl
import pandas as pd
import numpy as np

# Load local dataset
df = pd.read_csv("client_0.csv")  # Make sure to update this manually per client

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# Split dataset into train (90%) and test (10%)
split_idx = int(len(X) * 0.9)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Model setup
input_size = 2
hidden_size = 4
output_size = 1

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.weights_input_hidden = None
        self.weights_hidden_output = None

    def get_parameters(self, config):
        if self.weights_input_hidden is None or self.weights_hidden_output is None:
            print("[INFO] Initializing weights...")
            np.random.seed(0)
            self.weights_input_hidden = np.random.randn(input_size, hidden_size)
            self.weights_hidden_output = np.random.randn(hidden_size, output_size)

        return [self.weights_input_hidden, self.weights_hidden_output]

    def fit(self, parameters, config):
        self.weights_input_hidden, self.weights_hidden_output = map(np.array, parameters)
        print("weights_input_hidden:\n", self.weights_input_hidden)
        print("weights_hidden_output:\n", self.weights_hidden_output)
        print("-" * 50)
        self.train_model(X_train, y_train)
        return [self.weights_input_hidden, self.weights_hidden_output], len(X_train), {}

    def evaluate(self, parameters, config):
        self.weights_input_hidden, self.weights_hidden_output = map(np.array, parameters)

        hidden_input = np.dot(X_test, self.weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, self.weights_hidden_output)
        final_output = sigmoid(final_input)

        loss = np.mean((y_test - final_output) ** 2)
        return loss, len(X_test), {}

    def train_model(self, X, y):
        learning_rate = 0.01
        epochs = 10

        for epoch in range(epochs):
            hidden_input = np.dot(X, self.weights_input_hidden)
            hidden_output = sigmoid(hidden_input)
            final_input = np.dot(hidden_output, self.weights_hidden_output)
            final_output = sigmoid(final_input)

            error = y - final_output
            d_output = error * sigmoid_derivative(final_output)
            d_hidden = d_output.dot(self.weights_hidden_output.T) * sigmoid_derivative(hidden_output)

            self.weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
            self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate

            loss = np.mean(error ** 2)
            print(f" [Epoch {epoch + 1}/{epochs}] Loss: {loss:.6f}")

# Start client
fl.client.start_numpy_client(server_address="192.168.100.4:8080", client=FlowerClient())
