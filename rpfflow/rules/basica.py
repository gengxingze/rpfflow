import networkx as nx
from collections import Counter
from copy import deepcopy


# =====================================================
# 原子最大价电子表（可根据需要扩充）
# =====================================================
STANDARD_VALENCE = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "S": 6,
    "Cl": 1,
    "Cu": 2,
    "Pt": 6,
    "F": 1,
    "R": 1
}


# =====================================================
# 辅助函数：更新原子价态
# =====================================================
def update_valence(G: nx.Graph):
    """
    根据标准化学价与已成键数更新每个原子的价态信息。
    更新字段：
      - n_bonds：已成键数
      - expected_valence：标准价
      - valence：剩余可用价（expected_valence - n_bonds）
    """
    for node, data in G.nodes(data=True):
        symbol = data.get("symbol")
        n_bonds = 0
        if symbol not in STANDARD_VALENCE:
            raise ValueError(f"未知元素: {symbol}")
        for _, nbr, data in G.edges(node, data=True):
            n_bonds += data.get("bond_order", 1)
        expected_val = STANDARD_VALENCE[symbol]
        valence = expected_val - n_bonds

        G.nodes[node]["valence"] = valence


def check_valence_satisfied(G: nx.Graph, tolerance: int = 0):
    """
    检查所有原子是否满足标准价态。

    参数:
        G: 分子图（含有 "symbol" 属性）
        tolerance: 容许误差（默认为0，表示必须完全匹配）

    返回:
        (bool, dict)
        bool: 是否所有原子都满足标准价态
        dict: {节点索引: (当前价态, 最大价态)} 对不满足的原子进行记录
    """
    update_valence(G)

    violations = {}
    for node in G.nodes:
        sym = G.nodes[node]["symbol"]
        cur_val = G.nodes[node].get("valence", 0)
        max_val = G.nodes[node].get("max_valence", STANDARD_VALENCE.get(sym, 4))

        if abs(cur_val) > tolerance:
            violations[node] = (cur_val, max_val)

    # 若无违例，返回 True，否则 False
    return len(violations) == 0, violations


def check_element_conservation(G_reactant: nx.Graph, G_product: nx.Graph):
    """
    判断给定反应物图和生成物图是否满足元素守恒。

    参数:
        G_reactant: 反应物图（NetworkX）
        G_product: 生成物图（NetworkX）

    返回:
        (bool, dict)
        bool: 是否守恒
        dict: 不守恒的元素差异 {元素: (反应物数量, 生成物数量)}
    """
    # 统计元素个数
    elems_reactant = Counter(G_reactant.nodes[n]["symbol"] for n in G_reactant.nodes)
    elems_product = Counter(G_product.nodes[n]["symbol"] for n in G_product.nodes)

    # 找出不守恒的元素
    all_elems = set(elems_reactant.keys()) | set(elems_product.keys())
    diffs = {}
    for elem in all_elems:
        r_count = elems_reactant.get(elem, 0)
        p_count = elems_product.get(elem, 0)
        if r_count != p_count:
            diffs[elem] = (r_count, p_count)

    return len(diffs) == 0, diffs


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
        bond_order = new_G[atom1][atom2].get("bond_order")
        bond_order = bond_order - 1
        new_G[atom1][atom2]["bond_order"] = bond_order
        if bond_order <= 0:
            new_G.remove_edge(atom1, atom2)
        update_valence(new_G)
    return new_G


# =====================================================
# 成键操作
# =====================================================
def associate(G: nx.Graph, atom1: int, atom2: int, bond_order: float = 1.0, enforce: bool = False):
    """
    尝试在两个原子间成键。

    参数：
        G : nx.Graph
            分子图，节点包含“valence”与“n_bonds”等属性。
        atom1, atom2 : int
            要成键的原子索引。
        bond_order : float, default=1.0
            成键的键级（单键、双键等）。
        enforce : bool, default=False
            若为 True，则强制成键（忽略价态检查）。

    返回：
        new_G : nx.Graph
            成键后的新图。
    """
    new_G = deepcopy(G)
    update_valence(new_G)

    # 已存在的键处理逻辑
    def add_or_update_bond(graph, a1, a2, order):
        if graph.has_edge(a1, a2):
            existing = graph[a1][a2].get("bond_order", 1)
            graph[a1][a2]["bond_order"] = min(existing + order, 3)
        else:
            graph.add_edge(a1, a2, bond_order=order)

    if enforce:
        add_or_update_bond(new_G, atom1, atom2, bond_order)
    else:
        val1 = new_G.nodes[atom1].get("valence", 0) - bond_order
        val2 = new_G.nodes[atom2].get("valence", 0) - bond_order

        if val1 < 0 or val2 < 0:
            print(f"[Warning] Forming bond ({atom1}, {atom2}) with order {bond_order} "
                  f"would exceed the allowed valence. {new_G.nodes(data=True)}")
            return None
        else:
            add_or_update_bond(new_G, atom1, atom2, bond_order)

    update_valence(new_G)
    return new_G


# =====================================================
# 测试
# =====================================================
if __name__ == "__main__":
    from rpfflow.utils.convert import create_mol, rdkit_to_nx
    # 初始分子：甲酸 HCOOH
    smiles_start = "OC=O"
    mol_start = create_mol(smiles_start,add_h=True)
    G = rdkit_to_nx(mol_start)
    update_valence(G)

    # 检查原始分子价态
    ok, bad = check_valence_satisfied(G)
    print("是否满足标准价态:", ok)
    if not ok:
        print("不满足的原子:", bad)

    print("原始价态：", {n: G.nodes[n]["current_valence"] for n in G.nodes})

    # 尝试断开 C-H 键
    G2 = dissociate(G, 0, 1)
    print("断键后价态：", {n: G2.nodes[n]["current_valence"] for n in G2.nodes})

    # 尝试再成键 C-H
    G3 = associate(G2, 0, 1)
    print("成键后价态：", {n: G3.nodes[n]["current_valence"] for n in G3.nodes})

    print(G)

