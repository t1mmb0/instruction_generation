import networkx as nx
import gravis as gv
import matplotlib.pyplot as plt
import torch

class Graph:
    """
    A wrapper class for visualizing PyTorch Geometric data objects using NetworkX and Gravis.

    This class supports both standard `Data` objects and split objects created with
    `torch_geometric.transforms.RandomLinkSplit`. It constructs a NetworkX graph from either
    the structural edges (`edge_index`) or the candidate edges (`edge_label_index`).

    Attributes
    ----------
    G : nx.Graph
        The underlying NetworkX graph instance.
    num_nodes : int
        Number of nodes in the graph.

    Parameters
    ----------
    data : torch_geometric.data.Data
        A PyTorch Geometric data object, or a split object containing `edge_label_index`.
    use_labels : bool, optional (default: False)
        If True, the graph is constructed from `edge_label_index` (e.g., positive and negative
        candidate edges for link prediction). If False, the graph is constructed from
        `edge_index`.

    Methods
    -------
    visualize_graph(with_shell=True)
        Draws the graph with Matplotlib, showing a standard layout and an optional shell layout.
    visualize_interactive()
        Opens an interactive graph visualization in the browser using Gravis.
    """
    def __init__(self, data, use_labels=False):
        self.G = nx.Graph()

        # Edges 
        if hasattr(data, "edge_label_index") and use_labels:
            edges = data.edge_label_index.t().tolist()
        else:
            edges = data.edge_index.t().tolist()

        self.G.add_edges_from(edges)

        # Nodes 
        nodes = set([n for e in edges for n in e])
        self.G.add_nodes_from(nodes)

        # Optional: Features als Attribute
        if hasattr(data, "x"):
            for idx, feats in enumerate(data.x.tolist()):
                if idx in self.G.nodes:
                    self.G.nodes[idx]["features"] = feats

        self.num_nodes = self.G.number_of_nodes()

    def visualize_graph(self, with_shell=True):
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        nx.draw(self.G, with_labels=True, font_weight='bold')
        if with_shell:
            plt.subplot(122)
            nx.draw_shell(self.G, with_labels=True, font_weight='bold')
        plt.show()

    def visualize_interactive(self):
        fig = gv.vis(self.G)
        fig.display()

