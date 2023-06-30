import numpy as np
import networkx as nx
import seaborn as sns
import torchvision.models as models
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Create AlexNet model
alexnet = models.alexnet(pretrained=True)

# Interpret AlexNet as a DAG where each layer is a node
G_alexnet = nx.DiGraph()
nodes = list(alexnet.features) + list(alexnet.classifier)
G_alexnet.add_nodes_from(range(len(nodes)))
G_alexnet.add_edges_from((i, i+1) for i in range(len(nodes)-1))

# Get adjacency matrix of AlexNet
A_alexnet = nx.adjacency_matrix(G_alexnet).toarray()

# Use seaborn's heatmap function to visualize the adjacency matrix of AlexNet
plt.figure(figsize=(10, 10))
sns.heatmap(A_alexnet, annot=True, cmap='coolwarm', square=True, cbar=True, xticklabels=False, yticklabels=False)
plt.title('AlexNet Adjacency Matrix')
plt.show()

# Calculate the eigenvalues of the adjacency matrix
eigenvalues = np.linalg.eigvalsh(A_alexnet)

# Plot the spectral density
plt.figure(figsize=(10, 6))
plt.hist(eigenvalues, bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.title('Spectral Density')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.show()
