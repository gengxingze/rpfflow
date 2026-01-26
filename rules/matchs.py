import networkx as nx

def node_match(a, b):
    """节点匹配规则，可扩展（元素类型、电荷等）"""
    return a.get("symbol") == b.get("symbol")


def edge_match(a, b):
    """边匹配规则，可扩展（键级、键类型等）"""
    return a.get("bond_order", 1) == b.get("bond_order", 1)


def is_isomorphic(G1, G2):
    """
    判断两个图是否结构等价（同构）
    """
    GM = nx.isomorphism.GraphMatcher(G1, G2, node_match=node_match, edge_match=edge_match)
    return GM.is_isomorphic()


def is_subgraph(G_big, G_small):
    """
    判断 G_small 是否是 G_big 的子结构（子图同构）
    """
    GM = nx.isomorphism.GraphMatcher(G_big, G_small, node_match=node_match, edge_match=edge_match)
    return GM.subgraph_is_isomorphic()


def is_duplicate(G, graph_list):
    """
    判断图 G 是否与列表中已有图同构（去重）
    返回:
        bool: 是否重复
    """
    for Gi in graph_list:
        if is_isomorphic(G, Gi):
            return True
    return False


def match_target(G, target_list):
    """
    判断当前图是否与某个目标产物同构
    返回:
        (bool, target_index)
    """
    for i, target in enumerate(target_list):
        if is_isomorphic(G, target):
            return True, i
    return False, -1


if __name__ == "__main__":
    from graph_mm.molgraph import create_mol, rdkit_to_nx, draw_mol, create_common_molecules
    # === 1. 构建分子图 ===
    smiles_formic = "OC=O"     # 甲酸
    smiles_cooh = "C(=O)O"     # 羧基结构
    smiles_methanol = "CO"     # 甲醇

    mol_formic = create_mol(smiles_formic, add_h=True)
    mol_cooh = create_mol(smiles_cooh, add_h=True)
    mol_methanol = create_mol(smiles_methanol, add_h=True)

    draw_mol(mol_formic,filename="1.png")
    draw_mol(mol_cooh, filename="2.png")
    draw_mol(mol_methanol, filename="3.png")

    G_formic = rdkit_to_nx(mol_formic)
    G_cooh = rdkit_to_nx(mol_cooh)
    G_methanol = rdkit_to_nx(mol_methanol)

    module = create_common_molecules()

    print("✅ 已生成分子图：HCOOH, COOH, CH3OH")

    # === 2. 测试图同构 ===
    print("\n=== 同构测试 ===")
    print("HCOOH vs COOH 同构？", is_isomorphic(G_formic, G_cooh))
    print("HCOOH vs HCOOH 同构？", is_isomorphic(G_formic, G_formic))

    # === 3. 测试子图同构 ===
    from graph_mm.graph_ops import merge_graphs
    graph_list = [G_formic, G_cooh]
    graph_list = merge_graphs(graph_list)
    print("\n=== 子图同构测试 ===")
    print("COOH 是否为 合并图 子结构？", is_subgraph(G_formic, G_cooh))
    print("CH3OH 是否为 HCOOH 子结构？", is_subgraph(G_formic, G_methanol))

    # === 4. 测试重复判断 ===

    graph_list = [G_formic, G_cooh]
    print("\n=== 去重测试 ===")
    print("HCOOH 是否已存在？", is_duplicate(G_formic, graph_list))
    print("CH3OH 是否已存在？", is_duplicate(G_methanol, graph_list))

    # === 5. 测试目标匹配 ===
    target_list = [G_cooh, G_methanol]
    found, idx = match_target(G_formic, target_list)
    print("\n=== 目标匹配测试 ===")
    if found:
        print(f"HCOOH 与目标 {idx} 同构")
    else:
        print("HCOOH 未匹配任何目标")


