"""
mol_graph_utils.py

Utilities for:
- Creating RDKit molecules
- Generating 3D coordinates safely
- Converting between RDKit <-> NetworkX
- Converting RDKit -> ASE Atoms
- Building common molecular graph templates

Author: Xingze Geng
"""

from typing import  Optional
import networkx as nx
from rdkit import Chem
from ase import Atoms




# ============================================================
# RDKit <-> NetworkX conversion
# ============================================================

def rdkit_to_nx(mol: Chem.Mol) -> nx.Graph:
    """
    Convert RDKit Mol to NetworkX molecular graph.
    """
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), symbol=atom.GetSymbol())

    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_order=float(bond.GetBondTypeAsDouble()),
            bond_type=str(bond.GetBondType())
        )

    return G


def nx_to_rdkit(G: nx.Graph) -> Chem.Mol:
    """
    Convert NetworkX molecular graph to RDKit Mol.
    Requires:
        node: symbol
        edge: bond_order
    """
    mol = Chem.RWMol()
    node_map = {}

    for nid, attr in G.nodes(data=True):
        atom = Chem.Atom(attr.get("symbol", "C"))
        if "formal_charge" in attr:
            atom.SetFormalCharge(attr["formal_charge"])
        node_map[nid] = mol.AddAtom(atom)

    bond_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }

    for i, j, attr in G.edges(data=True):
        # 获取键阶并匹配到 RDKit 键类型
        bond_order = attr.get("bond_order")
        bond_type = bond_map.get(bond_order, Chem.BondType.SINGLE)

        # 添加化学键（若节点存在于映射中）
        if i in node_map and j in node_map:
            mol.AddBond(node_map[i], node_map[j], bond_type)

    return mol.GetMol()


# ============================================================
# RDKit -> ASE
# ============================================================
def rdkit_to_ase(mol: Chem.Mol, box_size: Optional[float] = None) -> Atoms:
    """
    Convert RDKit Mol to ASE Atoms (no hydrogen enforcement).
    """
    from rpfflow.core.structure import generate_coordinate
    mol = generate_coordinate(mol)
    conf = mol.GetConformer()

    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = [(conf.GetAtomPosition(i).x,
                  conf.GetAtomPosition(i).y,
                  conf.GetAtomPosition(i).z)
                 for i in range(mol.GetNumAtoms())]

    atoms = Atoms(symbols=symbols, positions=positions)

    if box_size is not None:
        atoms.set_cell([box_size] * 3)
        atoms.set_pbc(True)
        atoms.center()

    return atoms


def nx_to_ase(G: nx.Graph, box_size: Optional[float] = None) -> Atoms:
    """
    Convert NetworkX molecular graph directly to ASE Atoms.
    Pipeline: nx.Graph → RDKit Mol → ASE Atoms
    Requires:
        node: symbol
        edge: bond_order
    """
    mol = nx_to_rdkit(G)
    atoms = rdkit_to_ase(mol, box_size=box_size)
    return atoms




# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    from ase.io import write
    from rpfflow.utils.visualizer import draw_graph
    from rpfflow.core.structure import generate_coordinate, create_mol, create_common_molecules

    mol = create_mol("OC=O", add_h=True, optimize=True)
    G = rdkit_to_nx(mol)

    print(f"[✓] Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    draw_graph(G, save_path="formic_acid")

    module = create_common_molecules()

    ase_atoms = rdkit_to_ase(nx_to_rdkit(module["OCRO"]))
    write("mol.extxyz", ase_atoms)

    mol = generate_coordinate(nx_to_rdkit(module["OCRO"]))
    Chem.MolToMolFile(mol, "OCRO.mol")

    print("✓ All done.")
