"""
Plot multi-graphs in 3D.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import random

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection



params = {'legend.fontsize': 20,
'legend.handlelength': 50,
'figure.figsize':(20, 10)
}
plt.rcParams.update(params)



class LayeredNetworkGraph(object):

    def __init__(self, graphs, node_colors, node_labels=None, layout=nx.spring_layout, k=None, ax=None):
        """Given an ordered list of graphs [g1, g2, ..., gn] that represent
        different layers in a multi-layer network, plot the network in
        3D with the different layers separated along the z-axis.

        Within a layer, the corresponding graph defines the connectivity.
        Between layers, nodes in subsequent layers are connected if
        they have the same node ID.

        Arguments:
        ----------
        graphs : list of networkx.Graph objects
            List of graphs, one for each layer.

        node_labels : dict node ID : str label or None (default None)
            Dictionary mapping nodes to labels.
            If None is provided, nodes are not labelled.

        layout_func : function handle (default networkx.spring_layout)
            Function used to compute the layout.

        ax : mpl_toolkits.mplot3d.Axes3d instance or None (default None)
            The axis to plot to. If None is given, a new figure and a new axis are created.

        """

        # book-keeping
        self.graphs = graphs
        self.total_layers = len(graphs)

        self.node_labels = node_labels
        self.layout = layout
        self.k = k
        self.node_colors = node_colors

        if ax:
            self.ax = ax
        else:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')

        # create internal representation of nodes and edges
        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()

        # compute layout and plot
        self.get_node_positions(k=self.k)
        self.draw()

    

    def get_nodes(self):
        """Construct an internal representation of nodes with the format (node ID, layer)."""
        self.nodes = []
        for z, g in enumerate(self.graphs):
            self.nodes.extend([(node, z) for node in g.nodes()])


    def get_edges_within_layers(self):
        """Remap edges in the individual layers to the internal representations of the node IDs."""
        self.edges_within_layers = []
        for z, g in enumerate(self.graphs):
            self.edges_within_layers.extend([((source, z), (target, z)) for source, target in g.edges()])


    def get_node_positions(self, k=None, *args, **kwargs):
        """Get the node positions in the layered layout."""
        composition = self.graphs[0]
        for h in self.graphs[1:]:
            composition = nx.compose(composition, h)

        # Scale factor is proportional to the square root of the number of nodes
        num_nodes = composition.number_of_nodes()
        scale_factor = np.sqrt(num_nodes)

        pos = self.layout(composition, scale=scale_factor, k=k, *args, **kwargs)

        self.node_positions = dict()
        for z, g in enumerate(self.graphs):
            self.node_positions.update({(node, z) : (*pos[node], z) for node in g.nodes()})




    def draw_nodes(self, nodes, *args, **kwargs):
        x, y, z = zip(*[self.node_positions[node] for node in nodes])
        self.ax.scatter(x, y, z, *args, **kwargs)


    def draw_edges(self, edges, *args, **kwargs):
        segments = [(self.node_positions[source], self.node_positions[target]) for source, target in edges]
        line_collection = Line3DCollection(segments, *args, **kwargs)
        self.ax.add_collection3d(line_collection)


    def get_extent(self, pad=None):
        """Get the extent of the layout."""
        xyz = np.array(list(self.node_positions.values()))
        xmin, ymin, _ = np.min(xyz, axis=0)
        xmax, ymax, _ = np.max(xyz, axis=0)
        dx = xmax - xmin
        dy = ymax - ymin

        # Adjust padding based on the number of nodes
        if pad is None:
            pad = 0.1 * np.sqrt(len(self.nodes))

        return (xmin - pad * dx, xmax + pad * dx), \
               (ymin - pad * dy, ymax + pad * dy)

    
    
    def get_edges_between_layers(self):
        """Determine edges between layers. Nodes in subsequent layers are
        connected based on the edge list provided for inter-layer."""
        self.edges_between_layers = []
        nodes_g1 = list(self.graphs[0].nodes())
        nodes_g2 = list(self.graphs[1].nodes())
        num_edges = min(len(nodes_g1), len(nodes_g2))  # Or any other number you want
    
        for _ in range(num_edges):
            node1 = random.choice(nodes_g1)
            node2 = random.choice(nodes_g2)
            self.edges_between_layers.append(((node1, 0), (node2, 1)))
    
            # If you want to prevent the same node from being selected more than once, uncomment the following lines
            # nodes_g1.remove(node1)
            # nodes_g2.remove(node2)
    
    def get_interlayer_edges(graphs):
        num_layers = len(graphs)
        num_nodes = len(graphs[0].nodes)

        interlayer_edges = []
        for layer1 in range(num_layers):
            for layer2 in range(layer1+1, num_layers):
                for node in range(num_nodes):
                    interlayer_edges.append(((node, layer1), (node, layer2)))
        return interlayer_edges


    
    def draw_plane(self, z, *args, **kwargs):
        (xmin, xmax), (ymin, ymax) = self.get_extent(pad=0.1)
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u ,v)
        W = z * np.ones_like(U)

        #fig = plt.figure(figsize=(50, 50))

        self.ax.plot_surface(U, V, W, *args, **kwargs)
    


    def draw_node_labels(self, node_labels, *args, **kwargs):
        for node, z in self.nodes:
            if node in node_labels:
                ax.text(*self.node_positions[(node, z)], node_labels[node], *args, **kwargs)


    def draw(self):
        self.draw_edges(self.edges_within_layers,  color='k', alpha=0.3, linestyle='-', zorder=2)
        self.draw_edges(self.edges_between_layers, color='purple', alpha=0.5, linestyle=':', linewidth=2, zorder=2)

        for z in range(self.total_layers):
            self.draw_plane(z, alpha=0.2, zorder=1)
            layer_nodes = [node for node in self.nodes if node[1]==z]
            layer_colors = [self.node_colors[z][node[0]] for node in layer_nodes]
            self.draw_nodes(layer_nodes, c=layer_colors, s=90, zorder=3)

        if self.node_labels:
            self.draw_node_labels(self.node_labels,
                                  horizontalalignment='center',
                                  verticalalignment='center',
                                  zorder=100)


def calculate_supra_laplacian(graphs, inter_edges):
    # Calculate Laplacian for each layer
    L = [nx.laplacian_matrix(g).toarray() for g in graphs]

    # Construct supra-Laplacian
    num_layers = len(graphs)
    size = L[0].shape[0] # Assuming all graphs have the same size

    L_supra = np.zeros((size * num_layers, size * num_layers))

    for i, L_i in enumerate(L):
        # Diagonal blocks are the Laplacian matrices
        L_supra[i*size:(i+1)*size, i*size:(i+1)*size] = L_i

    # Inter-layer edges
    for edge in inter_edges:
        node1, layer1 = edge[0]
        node2, layer2 = edge[1]
        # Add inter-layer edge weights
        L_supra[layer1*size + node1, layer2*size + node2] = -1
        L_supra[layer2*size + node2, layer1*size + node1] = -1

    return L_supra



if __name__ == '__main__':
    # define graphs

    numnodes = 30
    numedge = 3

    # Create legend patches
    red_line = Line2D([0], [0], marker='o', color='w', label='Infected',
                  markerfacecolor='red', markersize=10)
    green_line = Line2D([0], [0], marker='o', color='w', label='Susceptible',
                    markerfacecolor='green', markersize=10)


    # Physical network structure
    G1 = nx.barabasi_albert_graph(numnodes, numedge)
    G2 = nx.barabasi_albert_graph(numnodes, numedge)

    # Diffusion model: SIR model
    model1 = ep.SIRModel(G1)
    model2 = ep.SIRModel(G2)

    config1 = mc.Configuration()
    config2 = mc.Configuration()
    config3 = mc.Configuration()
    config1.add_model_parameter('beta', 0.1)  # infection rate
    config1.add_model_parameter('gamma', 0)  # recovery rate
    config2.add_model_parameter('beta', 0.1)  # infection rate
    config2.add_model_parameter('gamma', 0)  # recovery rate
    config3.add_model_parameter('beta', 0.1)  # infection rate
    config3.add_model_parameter('gamma', 0)  # recovery rate

    # Initialize infection node
    initial_infected_node = 5
    for node in G1.nodes():
        if node == initial_infected_node:
            config1.add_node_configuration("status", node, 1)  # infected
        else:
            config1.add_node_configuration("status", node, 0)  # susceptible

    for node in G2.nodes():
        if node == initial_infected_node:
            config2.add_node_configuration("status", node, 1)  # infected
        else:
            config2.add_node_configuration("status", node, 0)  # susceptible


    model1.set_initial_status(config1)
    model2.set_initial_status(config2)


    # supra laplacian
    L1 = nx.laplacian_matrix(G1).toarray()
    L2 = nx.laplacian_matrix(G2).toarray()

    eigenvalues_L1 = np.linalg.eigvals(L1)
    eigenvalues_L2 = np.linalg.eigvals(L2)

    # Get the non-zero eigenvalues
    nonzero_eigenvalues_L1 = eigenvalues_L1[np.nonzero(eigenvalues_L1)]
    nonzero_eigenvalues_L2 = eigenvalues_L2[np.nonzero(eigenvalues_L2)]
    

    nonzero_eigenvalues_L1.sort()
    nonzero_eigenvalues_L2.sort()

    # Diffusion 
    D_values = np.linspace(0.1, 10, 100)

    D1 = 1
    D2 = 1

    lambda2_values_L1 = []
    lambda2_values_L2 = []
    lambda2_values_avg = []
    lambda_values_2Dx = []
    lambda2_values_supra = []

    for D in D_values:
        # Construct the supra-Laplacian matrix
        supra_L = np.block([[D1*L1 + D*np.eye(numnodes), -D * np.eye(numnodes)], [-D * np.eye(numnodes), D2*L2 + D*np.eye(numnodes)]])

        # Calculate the eigenvalues
        eigenvalues_supra = np.linalg.eigvals(supra_L)

        # Get the non-zero eigenvalues
        nonzero_eigenvalues_supra = eigenvalues_supra[np.nonzero(eigenvalues_supra)]

        # Sort the non-zero eigenvalues
        nonzero_eigenvalues_supra.sort()

        # Store the second smallest eigenvalues
        lambda2_values_L1.append(nonzero_eigenvalues_L1[1])
        lambda2_values_L2.append(nonzero_eigenvalues_L2[1])
        lambda2_values_avg.append((nonzero_eigenvalues_L1[1] + nonzero_eigenvalues_L2[1]) / 2)
        lambda_values_2Dx.append(2 * D)

        if len(nonzero_eigenvalues_supra) >= 2:
            lambda2_values_supra.append(nonzero_eigenvalues_supra[1])
        else:
            lambda2_values_supra.append(np.nan)  # Append a NaN if there are not enough non-zero eigenvalues

    # visualization
    pos1 = nx.spring_layout(G1, dim=3)
    pos2 = nx.spring_layout(G2, dim=3)

    for i in range(1):  # Run for 10 iterations
        model1.iteration()
        model2.iteration()

        # G1
        node_status1 = [model1.status[node] for node in range(numnodes)]
        colors1 = ['g' if s == 0 else 'r' if s == 1 else 'b' for s in node_status1]
        colors1[initial_infected_node] = 'k'

        # G2
        node_status2 = [model2.status[node] for node in range(numnodes)]
        colors2 = ['g' if s == 0 else 'r' if s == 1 else 'b' for s in node_status2]
        colors2[initial_infected_node] = 'k'

        node_labels = { nn : str(nn) for nn in range(numnodes)}


    # initialise figure and plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title(f"Iteration: {i+1}")
        ax.legend(handles=[red_line, green_line], frameon=False, prop={'size': 10}, handlelength=0.5)
        
        LayeredNetworkGraph([G1, G2], node_colors=[colors1, colors2], node_labels=node_labels, ax=ax, layout=nx.spring_layout, k=0.4)
        
        ax.set_axis_off()
        plt.show() 

    # supra vis
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
    #plt.legend()
    plt.show()

        