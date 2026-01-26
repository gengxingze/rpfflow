import networkx as nx
from copy import deepcopy


def dissociate(G: nx.Graph, atom1: int, atom2: int):
    """
    断开某个键，并更新边属性。

    参数:
        G: NetworkX 分子图
        atom1, atom2: 要断开的原子索引

    返回:
        新的 NetworkX 图对象
    """
    # 深拷贝，保证不会修改原图
    new_G = deepcopy(G)

    # 检查是否存在该键
    if new_G.has_edge(atom1, atom2):
        # 可以在这里加入化学规则约束，例如只允许断 C-H、O-H 等
        bond_order = new_G[atom1][atom2].get("bond_order")
        bond_order = bond_order - 1
        new_G[atom1][atom2]["bond_order"] = bond_order
        # 如果是单键或特定类型才断
        if bond_order <= 0:
            new_G.remove_edge(atom1, atom2)
    return new_G


def associate(G: nx.Graph, atom1: int, atom2: int):
    """
    形成新键，并更新边属性。

    参数:
        G: NetworkX 分子图
        atom1, atom2: 要成键的原子索引
        bond_order: 键级（1=单键，2=双键，3=三键）

    返回:
        新的 NetworkX 图对象
    """
    new_G = deepcopy(G)

    # 检查是否已经存在该键
    if new_G.has_edge(atom1, atom2):
        # 如果已存在，可以选择升级键级或忽略
        existing_order = new_G[atom1][atom2].get("bond_order", 1)
        new_order = existing_order + 1
        new_G[atom1][atom2]["bond_order"] = new_order
    else:
        new_G.add_edge(atom1, atom2, bond_order=1)

    return new_G
