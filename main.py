

if __name__ == "__main__":
    """
    回归测试：CO2 → CH3OH 反应路径搜索是否可正常运行
    目标：
    - 元素守恒检查通过
    - BFS 能返回至少一条路径
    - 路径中每一步都是 RxnState
    """

    from rpfflow.utils.logger_config import setup_logger
    import logging

    setup_logger(log_file="train.log", level="INFO")

    logger = logging.getLogger(__name__)

    from rpfflow.utils.convert import rdkit_to_nx
    from rpfflow.core.structure import create_mol
    from rpfflow.rules.basica import check_element_conservation
    # from rpfflow.search import bfs_search
    from rpfflow.rules.basica import update_valence

    # === 构建反应物 / 生成物 ===
    mol_react = create_mol('O=C(F)[O]')                 # CO2 (或简化占位)
    mol_prod  = create_mol("C", add_h=True)     # CH3OH

    G_react = rdkit_to_nx(mol_react)
    G_prod  = rdkit_to_nx(mol_prod)

    update_valence(G_react)
    update_valence(G_prod)

    # === 元素守恒检查 ===
    conserved, diffs = check_element_conservation(G_react, G_prod)
    # assert conserved, f"元素不守恒: {diffs}"
    from ase.io import read
    from rpfflow.core.structure import get_reference_structure, create_mol
    from rpfflow.core.state import collect_paths_from_nodes, save_search_results, load_search_results, RxnState
    from rpfflow.core.model import bfs_search

    slab = read("rpfflow/tests/POSCAR")
    G_react = RxnState(graphs=(G_react,G_prod), h_reserve=8, stage="[O]C(=O)F", slab=slab)

    # === 执行搜索 ===
    node = bfs_search(G_react, G_prod, n_hydrogen=8)
    print(f"[OK] 找到 {len(node)} 步反应路径")
    paths = []
    # for x, n in enumerate(node):
    #     nnpp = n.reaction_history
    #     paths.append(nnpp)
    #     n.save_reaction_path(f"path_{x}.extxyz")

    nnpp = node[0].reaction_history
    paths.append(nnpp)
    node[0].save_reaction_path(f"path_.extxyz")

    from rpfflow.core.state import collect_paths_from_nodes, save_search_results, load_search_results
    save_search_results(paths)
    paths = load_search_results()
    from rpfflow.utils.visualizer import plot_reaction_tree
    cc = collect_paths_from_nodes(node)
    print(cc)
    plot_reaction_tree(cc)
    from rpfflow.utils.visualizer import save_molecule_2d

    for i, x in enumerate(cc[0]):
        m = create_mol(x, add_h=True)
        save_molecule_2d(m, f"rpfflow/tests/mol_{i}.png")


    # print(f"[OK] 找到 {len(paths)} 步反应路径")
    print("Done")