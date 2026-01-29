"""
graph_ops.py

Utilities for:
- Splitting a graph into connected components
- Merging multiple graphs into one
- Simple demo for molecular graphs

Author: Xingze Geng
"""

from typing import List
import networkx as nx


# ============================================================
# Graph operations
# ============================================================

def split_graph(G: nx.Graph) -> List[nx.Graph]:
    """
    Split a graph into connected components.

    Args:
        G: input NetworkX graph

    Returns:
        List of subgraphs (deep copies).
    """
    subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    print(f"[INFO] Graph split into {len(subgraphs)} connected components.")
    return subgraphs


def merge_graphs(graphs: List[nx.Graph]) -> nx.Graph:
    """
    Merge multiple graphs into a single disjoint graph.

    Node indices are automatically relabeled.

    Args:
        graphs: list of NetworkX graphs

    Returns:
        Merged NetworkX graph.
    """
    G_merged = nx.disjoint_union_all(graphs)
    print(
        f"[INFO] Merged {len(graphs)} graphs | "
        f"Nodes: {G_merged.number_of_nodes()} | "
        f"Edges: {G_merged.number_of_edges()}"
    )
    return G_merged


# ============================================================
# Example usage
# ============================================================

def demo():
    """
    Demo: build molecular graphs, merge and split them.
    """
    from convert import rdkit_to_nx
    from visualizer import plot_molecular_graph
    from rpfflow.core.structure import create_mol

    mol1 = create_mol("OC=O", add_h=True)   # HCOOH
    mol2 = create_mol("C=O", add_h=True)    # H2CO

    G1 = rdkit_to_nx(mol1)
    G2 = rdkit_to_nx(mol2)

    print(f"[INFO] Graph 1: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
    print(f"[INFO] Graph 2: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")

    plot_molecular_graph(G1, title="Molecule 1")
    plot_molecular_graph(G2, title="Molecule 2")

    # Merge
    G_merge = merge_graphs([G1, G2])
    plot_molecular_graph(G_merge, title="Merged graph")

    # Split
    subgraphs = split_graph(G_merge)
    for i, g in enumerate(subgraphs):
        plot_molecular_graph(g, title=f"Subgraph {i}")


if __name__ == "__main__":
    demo()
