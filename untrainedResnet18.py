import numpy as np
import networkx as nx
import seaborn as sns
import torchvision.models as models
import matplotlib.pyplot as plt
import os
import torch
import torchvision
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Create model
resnet18 = models.resnet18(pretrained=False)

# Interpret as a DAG where each layer is a node
G_resnet18 = nx.DiGraph()

# list out the layers in order
nodes = [resnet18.conv1, resnet18.bn1, resnet18.relu, resnet18.maxpool]

# Function to add nodes and edges from layers
def add_layer_connections(layer, start_node=0):
    global nodes
    if isinstance(layer, torch.nn.modules.container.Sequential):
        for l in layer:
            add_layer_connections(l, start_node=len(nodes))
    elif isinstance(layer, torchvision.models.resnet.BasicBlock):
        nodes += [layer.conv1, layer.conv2]
        G_resnet18.add_edges_from([(start_node, len(nodes)-2), (len(nodes)-2, len(nodes)-1)])
        G_resnet18.add_edge(start_node, len(nodes)-1)  # skip connection
    else:
        nodes.append(layer)

# Add layers and connections for ResNet
add_layer_connections(resnet18.layer1)
add_layer_connections(resnet18.layer2)
add_layer_connections(resnet18.layer3)
add_layer_connections(resnet18.layer4)
nodes += [resnet18.avgpool, resnet18.fc]

G_resnet18.add_nodes_from(range(len(nodes)))
for i in range(4, len(nodes)-1):  # Connect subsequent layers
    G_resnet18.add_edge(i, i+1)

# Get adjacency matrix
A_resnet18 = nx.adjacency_matrix(G_resnet18).toarray()

# Use seaborn's heatmap function to visualize the adjacency matrix
plt.figure(figsize=(10, 10))
sns.heatmap(A_resnet18, annot=True, cmap='coolwarm', square=True, cbar=True, xticklabels=False, yticklabels=False)
plt.title('ResNet18 Adjacency Matrix')
plt.show()


# Calculate the eigenvalues of the adjacency matrix
eigenvalues = np.linalg.eigvalsh(A_resnet18)

# Plot the spectral density
plt.figure(figsize=(10, 6))
plt.hist(eigenvalues, bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.title('Spectral Density')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.show()