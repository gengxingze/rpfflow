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

from copy import deepcopy
from typing import Dict, Optional

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms


# ============================================================
# RDKit molecule construction
# ============================================================

def create_mol(
    smiles: Optional[str] = None,
    structure: Optional[str] = None,
    add_h: bool = False,
    optimize: bool = False
) -> Chem.Mol:
    """
    Create an RDKit molecule from SMILES or structure file.

    Args:
        smiles: SMILES string
        structure: mol file path
        add_h: whether to add hydrogens
        optimize: whether to run UFF optimization

    Returns:
        RDKit Mol object with 3D conformer.
    """
    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
    elif structure is not None:
        mol = Chem.MolFromMolFile(structure)
    else:
        raise ValueError("Either smiles or structure must be provided.")

    if mol is None:
        raise ValueError("Failed to create RDKit Mol.")

    if add_h:
        mol = Chem.AddHs(mol)

    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL, catchErrors=True)

    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if optimize:
        AllChem.UFFOptimizeMolecule(mol)

    return mol


def generate_coordinate(mol: Chem.Mol) -> Chem.Mol:
    """
    Safely generate 3D coordinates without enforcing full sanitization.
    Suitable for TS or broken structures.
    """
    if not isinstance(mol, Chem.Mol):
        raise TypeError("Input must be an RDKit Mol object.")

    mol = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SANITIZE_SETAROMATICITY |
                        Chem.SANITIZE_FINDRADICALS |
                        Chem.SANITIZE_SETCONJUGATION
        )
    except Exception as e:
        print("[WARN] Partial sanitize failed (allowed):", e)
        mol.UpdatePropertyCache(strict=False)

    if mol.GetNumConformers() == 0:
        params = AllChem.ETKDG()
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError("3D embedding failed.")

    return mol


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
# Common molecular graph templates
# ============================================================
def create_common_molecules() -> Dict[str, nx.Graph]:
    """
    Build common molecular graph templates.
    """
    molecules = {}

    smiles_dict = {
        "OH_": "[OH-]",
        "H": "[H]",
        "H2O": "O",
        "CO2": "O=C=O",
    }

    for name, smi in smiles_dict.items():
        mol = create_mol(smi, add_h=True, optimize=True)
        molecules[name] = rdkit_to_nx(mol)
        print(f"[✓] {name} generated.")

    from rules.basica import update_valence

    # O=C-O-R
    G1 = nx.Graph()
    G1.add_nodes_from([
        (0, {"symbol": "O"}),
        (1, {"symbol": "C"}),
        (2, {"symbol": "O"}),
        (3, {"symbol": "F"}),  # active site
    ])
    G1.add_edge(0, 1, bond_order=2.0)
    G1.add_edge(1, 2, bond_order=1.0)
    G1.add_edge(2, 3, bond_order=1.0)
    update_valence(G1)
    molecules["OCOR"] = G1

    # O=C(R)-O
    G2 = nx.Graph()
    G2.add_nodes_from([
        (0, {"symbol": "O"}),
        (1, {"symbol": "C"}),
        (2, {"symbol": "O"}),
        (3, {"symbol": "F"}),
    ])
    G2.add_edge(0, 1, bond_order=2.0)
    G2.add_edge(1, 2, bond_order=1.0)
    G2.add_edge(1, 3, bond_order=1.0)
    update_valence(G2)
    molecules["OCRO"] = G2

    R = deepcopy(molecules["H"])
    R.nodes[0]["symbol"] = "F"
    update_valence(R)
    molecules["F"] = R

    return molecules


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    from ase.io import write
    from graph_mm.visualizer import draw_graph

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
