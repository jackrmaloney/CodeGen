import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Create the synthetic dataset
x = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(x)

# Reshape data for PyTorch
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Convert to PyTorch tensors
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define a 3-layer MLP
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Save the initial weights
initial_weights = [w.clone() for w in model.parameters()]

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(50):
    # Forward pass
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print MSE and RMSE
    print(f"Epoch {epoch+1}, MSE: {loss.item()}, RMSE: {np.sqrt(loss.item())}")

# Save the final weights
final_weights = [w.clone() for w in model.parameters()]

# Function to flatten weights for comparison
def flatten_weights(weights):
    return torch.cat([w.view(-1) for w in weights])

# Flatten initial and final weights
initial_weights_flat = flatten_weights(initial_weights)
final_weights_flat = flatten_weights(final_weights)

# Print out some of the initial and final weights
print("Initial weights: ", initial_weights_flat[:8])
print("Final weights: ", final_weights_flat[:8])

# After training, compute the R2 Score
model.eval()
with torch.no_grad():
    y_pred = model(x)
r2 = r2_score(y.numpy(), y_pred.numpy())
print("R2 Score: ", r2)

# Convert the predictions to numpy array for plotting
y_pred_np = y_pred.detach().numpy()

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='True')
plt.plot(x, y_pred_np, label='Predicted')
plt.legend()
plt.show()
