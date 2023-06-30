import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt


N = 20
k_avg = 8
m = int(k_avg / 2)

G1 = nx.barabasi_albert_graph(N, m)
G2 = nx.erdos_renyi_graph(N, k_avg / (N - 1))

A1 = nx.adjacency_matrix(G1).toarray()
A2 = nx.adjacency_matrix(G2).toarray()

eigenvalues_A1 = np.linalg.eigvals(A1)
eigenvalues_A2 = np.linalg.eigvals(A2)

eigenvalues_A1.sort()
eigenvalues_A2.sort()

# Diffusion 
D_values = np.linspace(0.1, 10, 100)

lambda2_values_A1 = []
lambda2_values_A2 = []
lambda2_values_avg = []
lambda_values_2Dx = []
lambda2_values_supra = []

D1 = 1
D2 = 1

for D in D_values:
    # Construct the supra adjacency matrix
    supra_A = np.block([[A1, np.zeros(A1.shape)], [np.zeros(A2.shape), A2]])
    
    eigenvalues_supra = np.linalg.eigvals(supra_A)
    
    eigenvalues_supra.sort()
    
    lambda2_values_A1.append(eigenvalues_A1[1])
    lambda2_values_A2.append(eigenvalues_A2[1])
    lambda2_values_avg.append((eigenvalues_A1[1] + eigenvalues_A2[1]) / 2)
    lambda_values_2Dx.append(2 * D)
    lambda2_values_supra.append(eigenvalues_supra[1])

plt.figure(figsize=(8, 8))
sns.heatmap(supra_A, annot=True, cmap='coolwarm', square=True, cbar=True, xticklabels=False, yticklabels=False)
plt.title('Supra Adjacency Matrix')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(D_values, lambda2_values_A1, label='λ2 of L1')
plt.plot(D_values, lambda2_values_A2, label='λ2 of L2')
plt.plot(D_values, lambda2_values_avg, label='λ2 of (L1+L2)/2')
plt.plot(D_values, lambda_values_2Dx, label='2Dx')
plt.plot(D_values, lambda2_values_supra, label='λ2 of Supra-Laplacian')
plt.xlabel('Interlayer diffusion coefficient (D)')
plt.ylabel('Second smallest eigenvalue')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

