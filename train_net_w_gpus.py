"""
2023-11-30 

Simple feedforward neural network. 

Intial prototype by ChatGPT 4. Modified by Katie Keith 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Check if there 
if torch.cuda.is_available(): #For NVIDIA CUDA devices 
    device = torch.device('cuda')
    print("GPUs: Using CUDA device")

elif torch.backends.mps.is_available(): #For Apple M1, M2 chips
    device = torch.device('mps')
    print("GPUs: Using MPS device.")
    
else:
    device = torch.device('cpu')
    print("No GPUs available, using CPU.")

# Define the neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 5)  # 2 input features, 5 neurons in the first hidden layer
        self.fc2 = nn.Linear(5, 3)  # 5 neurons in the first hidden layer, 3 neurons in the second hidden layer
        self.fc3 = nn.Linear(3, 1)  # 3 neurons in the second hidden layer, 1 output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the neural network and move it to the GPU if available
net = SimpleNet().to(device)

# Generate some fake data and move it to the GPU if available
rng = np.random.default_rng(0)
X = rng.random((100, 2))  # 100 samples, 2 features each
y = (X[:, 0] + X[:, 1] > 1).astype(float)  # Simple threshold to create binary labels

# Convert data to PyTorch tensors and move them to the GPU if available
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # For binary classification
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = net(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
