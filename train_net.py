"""
2023-11-30 

Simple feedforward neural network. 

Intial prototype by ChatGPT 4. Modified by Katie Keith 
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Creating a simple feedforward neural network using PyTorch

# Define the neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 5) # 2 input features, 5 neurons in the first hidden layer
        self.fc2 = nn.Linear(5, 3) # 5 neurons in the first hidden layer, 3 neurons in the second hidden layer
        self.fc3 = nn.Linear(3, 1) # 3 neurons in the second hidden layer, 1 output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the neural network
net = SimpleNet()

# Generate some fake data
rng = np.random.default_rng(0)
X = rng.random((100, 2))  # 100 samples, 2 features each
y = (X[:, 0] + X[:, 1] > 1).astype(float)  # simple logic to create labels

# Convert numpy arrays to torch tensors
X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32).view(-1, 1)

print("X[0:3] =", X_torch[0:3])
print("y[0:3] =",y_torch[0:3])

# Define a loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Training the network
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = net(X_torch)
    loss = criterion(outputs, y_torch)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 9:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# SimpleNet, fake data, and training process ready
print("Simple feedforward neural network and training process completed.")