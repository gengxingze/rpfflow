from rdkit import Chem


def graph_to_smiles(G):
    # 创建一个可编辑的 RDKit 分子
    mol = Chem.RWMol()

    # 1. 添加原子
    node_to_idx = {}
    for n, data in G.nodes(data=True):
        symbol = data["symbol"]
        # 如果 F 代表表面，在 SMILES 中通常建议用方括号 [F] 明确
        atom = Chem.Atom(symbol)
        idx = mol.AddAtom(atom)
        node_to_idx[n] = idx

    # 2. 添加键
    rdkit_bond_types = {1.0: Chem.BondType.SINGLE, 2.0: Chem.BondType.DOUBLE, 3.0: Chem.BondType.TRIPLE}
    for u, v, data in G.edges(data=True):
        order = data.get("bond_order", 1.0)
        mol.AddBond(node_to_idx[u], node_to_idx[v], rdkit_bond_types.get(order, Chem.BondType.SINGLE))

    # 3. 转换为标准 SMILES
    return Chem.MolToSmiles(mol.GetMol())



import networkx as nx
from rpfflow.rules.basica import update_valence
# O-C-O-R
G1 = nx.Graph()
G1.add_nodes_from([
    (0, {"symbol": "O"}),
    (1, {"symbol": "C"}),
    (2, {"symbol": "O"}),
    (3, {"symbol": "F"}),  # active site
])
G1.add_edge(0, 1, bond_order=2.0)
G1.add_edge(1, 3, bond_order=1.0)
G1.add_edge(1, 2, bond_order=1.0)
update_valence(G1)
# 测试您的 G1
from rpfflow.utils.convert import nx_to_rdkit
# 假设 mol 是由 nx_to_rdkit(G1) 生成的
mol = nx_to_rdkit(G1)

# for atom in mol.GetAtoms():
#     # 找到那个碳原子 (假设索引为 1)
#     if atom.GetSymbol() == "C":
#         # 计算当前已有的键总数 (1.0 + 1.0 = 2)
#         # 如果你希望它是一个单电子自由基，且带有一个隐式氢：
#         # 4 (价电子) - 2 (化学键) - 1 (隐式氢) = 1 (自由基电子)
#         atom.SetNumRadicalElectrons(1)
#     if atom.GetSymbol() == "O":
#         # 计算当前已有的键总数 (1.0 + 1.0 = 2)
#         # 如果你希望它是一个单电子自由基，且带有一个隐式氢：
#         # 4 (价电子) - 2 (化学键) - 1 (隐式氢) = 1 (自由基电子)
#         atom.SetNumRadicalElectrons(1)

# 再次生成 SMILES
print(Chem.MolToSmiles(mol))
from rpfflow.utils.visualizer import save_molecule_2d
from rpfflow.core.structure import create_mol
mol = create_mol("[H]OC(=O)F.C")
save_molecule_2d(mol, filename="test.png")
# print(f"生成的 SMILES: {smiles_str}")
# 输出应该是: O=CO[F]

