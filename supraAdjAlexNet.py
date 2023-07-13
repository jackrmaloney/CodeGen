import numpy as np
import networkx as nx
import seaborn as sns
import torchvision.models as models
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

N = 5  # Number of nodes in each layer
m = 2  # Number of edges to attach from a new node to existing nodes

# Generate two different networks
G1 = nx.barabasi_albert_graph(N, m)
G2 = nx.erdos_renyi_graph(N, m)

# Create AlexNet model
alexnet = models.alexnet()

# Interpret AlexNet as a DAG where each layer is a node
G_alexnet = nx.DiGraph()
nodes = list(alexnet.features) + list(alexnet.classifier)
G_alexnet.add_nodes_from(range(len(nodes)))
G_alexnet.add_edges_from((i, i+1) for i in range(len(nodes)-1))

# Get adjacency matrix of AlexNet
A_alexnet = nx.adjacency_matrix(G_alexnet).toarray()

# Create zero matrices of appropriate size for G1 and G2
A1_large = np.zeros(A_alexnet.shape)
A2_large = np.zeros(A_alexnet.shape)

# Get adjacency matrices of these networks
A1 = nx.adjacency_matrix(G1).toarray()
A2 = nx.adjacency_matrix(G2).toarray()

# Place the original adjacency matrices in the top left corner of the new ones
A1_large[:A1.shape[0], :A1.shape[1]] = A1
A2_large[:A2.shape[0], :A2.shape[1]] = A2

# Construct the supra adjacency matrix
supra_A = np.block([[A1_large, np.zeros(A1_large.shape)], [np.zeros(A2_large.shape), A2_large]])

# Use seaborn's heatmap function to visualize the supra adjacency matrix
plt.figure(figsize=(16, 16))
sns.heatmap(supra_A, annot=True, cmap='coolwarm', square=True, cbar=True, xticklabels=False, yticklabels=False)
plt.title('Supra Adjacency Matrix')
plt.show()

# Calculate the eigenvalues of the supra adjacency matrix
eigenvalues = np.linalg.eigvalsh(supra_A)

# Plot the spectral density
plt.figure(figsize=(10, 6))
plt.hist(eigenvalues, bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.title('Spectral Density')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.show()
