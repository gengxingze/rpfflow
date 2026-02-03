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
from rpfflow.rules.basica import update_valence



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
    将 NetworkX 分子图转换为 RDKit Mol，并根据剩余价态自动设置自由基。
    """
    # 1. 首先调用更新价态的辅助函数
    # 确保每个节点都有了 "valence" 属性
    update_valence(G)

    mol = Chem.RWMol()
    node_map = {}

    # 2. 添加原子并处理自由基
    for nid, attr in G.nodes(data=True):
        symbol = attr.get("symbol", "C")
        atom = Chem.Atom(symbol)

        # 处理电荷
        if "formal_charge" in attr:
            atom.SetFormalCharge(attr["formal_charge"])

        # --- 核心逻辑：处理自由基 ---
        # 获取 update_valence 计算出的剩余价态
        rem_valence = attr.get("valence", 0)

        if rem_valence > 0:
            # 如果剩余价态 > 0，说明存在未成对电子
            # 设置自由基电子数。通常为 1，也可以设为 rem_valence
            atom.SetNumRadicalElectrons(int(rem_valence))

            # 为了防止 RDKit 自动补氢把自由基“抹除”
            # 我们显式关闭该原子的隐式氢处理
            atom.SetNoImplicit(True)
        # --------------------------

        node_map[nid] = mol.AddAtom(atom)

    # 3. 添加化学键
    bond_map = {
        1.0: Chem.BondType.SINGLE,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
    }

    for i, j, attr in G.edges(data=True):
        bond_order = attr.get("bond_order", 1.0)
        bond_type = bond_map.get(bond_order, Chem.BondType.SINGLE)

        if i in node_map and j in node_map:
            mol.AddBond(node_map[i], node_map[j], bond_type)

    # 4. 转换回普通的 Mol 对象
    final_mol = mol.GetMol()

    # 最后进行简单的检查，但不进行严格的 Sanitize（因为自由基可能不符合标准价键规则）
    final_mol.UpdatePropertyCache(strict=False)

    return final_mol


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
