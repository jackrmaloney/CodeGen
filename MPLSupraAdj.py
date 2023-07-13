import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(x)

# Reshape data
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Convert to PyTorch tensors
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# MLP
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(500):
    # Forward pass
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, MSE: {loss.item()}, RMSE: {np.sqrt(loss.item())}")

# R2 Score
model.eval()
with torch.no_grad():
    y_pred = model(x)
r2 = r2_score(y.numpy(), y_pred.numpy())
print("R2 Score: ", r2)

y_pred_np = y_pred.detach().numpy()

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y.numpy(), label='True')
plt.plot(x.numpy(), y_pred_np, label='Predicted')
plt.legend()
plt.show()

# Save the final weights
final_weights = [w.clone() for w in model.parameters()]

# Extract adjacency matrices (i.e., weights between layers)
adjacency_matrices = [w.detach().numpy() for w in final_weights if len(w.shape) == 2]

# Get the dimensions
max_rows = max(adj.shape[0] for adj in adjacency_matrices)
max_cols = max(adj.shape[1] for adj in adjacency_matrices)

# Pad smaller adjacency matrices with zeros
padded_adjacency_matrices = [np.pad(adj, ((0, max_rows - adj.shape[0]), (0, max_cols - adj.shape[1]))) for adj in adjacency_matrices]

# Stack adjacency matrices into supra adjacency matrix
supra_adjacency_matrix = np.block([
    [adj if i==j else np.zeros_like(adj) for j, adj in enumerate(padded_adjacency_matrices)]
    for i, adj in enumerate(padded_adjacency_matrices)
])

# Plot supra adjacency matrix
plt.figure(figsize=(10, 10))
sns.heatmap(supra_adjacency_matrix, square=True, cmap='coolwarm')
plt.title('Supra Adjacency Matrix')
plt.show()

eigenvalues = np.linalg.eigvalsh(supra_adjacency_matrix)

# Plot the spectral density
plt.figure(figsize=(10, 6))
plt.hist(eigenvalues, bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.title('Spectral Density')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.show()

