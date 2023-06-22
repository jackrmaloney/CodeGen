import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


N = 100
k_avg = 8
m = int(k_avg / 2)

G1 = nx.barabasi_albert_graph(N, m)
G2 = nx.erdos_renyi_graph(N, k_avg / (N - 1))

L1 = nx.laplacian_matrix(G1).toarray()
L2 = nx.laplacian_matrix(G2).toarray()

eigenvalues_L1 = np.linalg.eigvals(L1)
eigenvalues_L2 = np.linalg.eigvals(L2)

eigenvalues_L1.sort()
eigenvalues_L2.sort()

# Diffusion 
D_values = np.linspace(0.1, 10, 100)

lambda2_values_L1 = []
lambda2_values_L2 = []
lambda2_values_avg = []
lambda_values_2Dx = []
lambda2_values_supra = []

D1 = 1
D2 = 1

for D in D_values:
    # Construct the supra-Laplacian matrix
    supra_L = np.block([[D1*L1 + D*np.eye(N), -D * np.eye(N)], [-D * np.eye(N), D2*L2 + D*np.eye(N)]])

    eigenvalues_supra = np.linalg.eigvals(supra_L)
    
    eigenvalues_supra.sort()
    
    lambda2_values_L1.append(eigenvalues_L1[1])
    lambda2_values_L2.append(eigenvalues_L2[1])
    lambda2_values_avg.append((eigenvalues_L1[1] + eigenvalues_L2[1]) / 2)
    lambda_values_2Dx.append(2 * D)
    lambda2_values_supra.append(eigenvalues_supra[1])

plt.figure(figsize=(10, 6))
plt.plot(D_values, lambda2_values_L1, label='λ2 of L1')
plt.plot(D_values, lambda2_values_L2, label='λ2 of L2')
plt.plot(D_values, lambda2_values_avg, label='λ2 of (L1+L2)/2')
plt.plot(D_values, lambda_values_2Dx, label='2Dx')
plt.plot(D_values, lambda2_values_supra, label='λ2 of Supra-Laplacian')
plt.xlabel('Interlayer diffusion coefficient (D)')
plt.ylabel('Second smallest eigenvalue')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
