"""
mol_visualization.py

Visualization utilities for:
- NetworkX molecular graphs
- RDKit 2D molecules
- RDKit 3D molecules (py3Dmol)

Author: Xingze Geng
"""

from typing import List, Optional, Callable

import py3Dmol
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


# ============================================================
# Visualization configuration
# ============================================================

ELEMENT_COLORS = {
    "H": "#CCCCCC",
    "C": "#4C4C4C",
    "O": "#E74C3C",
    "N": "#3498DB",
    "S": "#F1C40F",
    "P": "#9B59B6",
    "F": "#1ABC9C",  # virtual / active site
}


# ============================================================
# NetworkX molecular graph visualization
# ============================================================

def plot_molecular_graph(
    G: nx.Graph,
    title: str = "Molecular Graph",
    save_path: Optional[str] = None,
    layout: Callable = nx.spring_layout,
    node_size: int = 800,
    font_size: int = 12,
) -> None:
    """
    Plot a molecular graph using NetworkX + Matplotlib.

    Required node attribute:
        - symbol

    Optional edge attribute:
        - bond_order
    """
    plt.rcParams["font.family"] = "Times New Roman"

    try:
        node_colors = [
            ELEMENT_COLORS.get(G.nodes[n].get("symbol", "R"), "#BDC3C7")
            for n in G.nodes
        ]

        labels = {n: G.nodes[n].get("symbol", str(n)) for n in G.nodes}
        pos = layout(G, seed=159632)

        plt.figure(figsize=(6, 6))
        nx.draw(
            G,
            pos,
            labels=labels,
            node_color=node_colors,
            node_size=node_size,
            font_size=font_size,
            with_labels=True,
            edgecolors="black",
            linewidths=1.2,
            width=1.6,
        )

        edge_labels = {
            (u, v): d.get("bond_order", "")
            for u, v, d in G.edges(data=True)
        }

        if edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, font_size=font_size - 2
            )

        plt.title(title)
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        plt.show()

    except Exception as e:
        plt.close()
        raise RuntimeError(f"Graph visualization failed: {e}")


def plot_molecular_graphs(
    graphs: List[nx.Graph],
    titles: Optional[List[str]] = None,
    cols: int = 3,
    save_path: Optional[str] = None,
    layout: Callable = nx.spring_layout,
    node_size: int = 800,
    font_size: int = 12,
) -> None:
    """
    Plot multiple molecular graphs in a grid layout.
    """
    plt.rcParams["font.family"] = "Times New Roman"

    n = len(graphs)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
    axes = np.atleast_2d(axes)

    for i, G in enumerate(graphs):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.set_axis_off()

        node_colors = [
            ELEMENT_COLORS.get(G.nodes[n].get("symbol", "R"), "#BDC3C7")
            for n in G.nodes
        ]

        labels = {n: G.nodes[n].get("symbol", str(n)) for n in G.nodes}
        pos = layout(G, seed=159632)

        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            labels=labels,
            node_color=node_colors,
            node_size=node_size,
            font_size=font_size,
            edgecolors="black",
            linewidths=1.2,
            width=1.6,
        )

        edge_labels = {
            (u, v): d.get("bond_order", "")
            for u, v, d in G.edges(data=True)
        }

        if edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, font_size=font_size - 2, ax=ax
            )

        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=font_size + 2)
        else:
            ax.set_title(f"Molecule {i+1}", fontsize=font_size + 2)

    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


# ============================================================
# RDKit molecule visualization
# ============================================================

def show_molecule_3d(
    mol: Chem.Mol,
    style: str = "stick",
    width: int = 400,
    height: int = 400,
    export_html: Optional[str] = None,
):
    """
    Visualize an RDKit Mol in 3D using py3Dmol.
    """
    mol_block = Chem.MolToMolBlock(Chem.Mol(mol))

    view = py3Dmol.view(width=width, height=height)
    view.addModel(mol_block, "mol")
    view.setStyle({style: {}})
    view.zoomTo()
    view.show()

    if export_html:
        html = view._make_html()
        with open(export_html, "w") as f:
            f.write(html)

    return view


def save_molecule_2d(
    mol: Chem.Mol,
    filename: str = "molecule.png",
    size=(500, 500),
) -> None:
    """
    Save 2D depiction of an RDKit molecule.
    """
    # mol 原本只有 C 和 O
    # mol = Chem.RemoveHs(mol)  # 移除所有显式氢（画图时不会自动补）
    drawer = Draw.MolDraw2DCairo(*size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    with open(filename, "wb") as f:
        f.write(drawer.GetDrawingText())


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    from convert import (
        create_mol,
        rdkit_to_nx,
        generate_coordinate,
        create_common_molecules,
        nx_to_rdkit,
    )

    mol = create_mol("O=C(O)C(=O)O", add_h=False)
    G = rdkit_to_nx(mol)
    from rpfflow.core.state import GraphState
    G = GraphState(graph=[G])
    G.update()
    plot_molecular_graph(G.graph[0], title="Formic acid", save_path="formic_graph.png")

    templates = create_common_molecules()
    mol_tmp = generate_coordinate(nx_to_rdkit(templates["OCRO"]))
    save_molecule_2d(mol_tmp)
    save_molecule_2d(mol, filename="../../graph_mm/test.png")

    print("✓ Visualization done.")
