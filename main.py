import logging
import os
from ase.io import read
from rpfflow.utils.logger_config import setup_logger
from rpfflow.utils.convert import rdkit_to_nx
from rpfflow.core.structure import create_mol
from rpfflow.rules.basica import check_element_conservation, update_valence
from rpfflow.core.state import RxnState, collect_paths_from_nodes, save_search_results, load_search_results
from rpfflow.core.model import bfs_search
from rpfflow.utils.visualizer import plot_reaction_tree, save_molecule_2d, draw_reaction_networks


def main():
    # === 1. 环境与日志配置 ===
    setup_logger(log_file="regression_test.log", level="INFO")
    logger = logging.getLogger(__name__)
    logger.info("Starting regression test: CO2 to CH3OH reaction path search.")

    # === 2. 构建反应物与产物图结构 ===
    # 示例：使用 [O]C(=O)F 作为起始状态，C (Methane) 或 CH3OH 作为目标
    smiles_react = '[O]C(=O)F'
    smiles_prod = 'C'

    mol_react = create_mol(smiles_react, add_h=True)
    mol_prod = create_mol(smiles_prod, add_h=True)

    G_react_nx = rdkit_to_nx(mol_react)
    G_prod_nx = rdkit_to_nx(mol_prod)
    # 更新价键信息，确保搜索规则识别准确
    update_valence(G_react_nx)
    update_valence(G_prod_nx)

    # === 3. 元素守恒检查 ===
    conserved, diffs = check_element_conservation(G_react_nx, G_prod_nx)
    if not conserved:
        logger.warning(f"Element imbalance detected: {diffs}. Ensure h_reserve covers the gap.")

    # === 4. 初始化起始状态 (RxnState) ===
    # 加载 Slab 结构
    poscar_path = "rpfflow/tests/Cu.xyz"
    if not os.path.exists(poscar_path):
        raise FileNotFoundError(f"Missing POSCAR at {poscar_path}")

    slab = read(poscar_path)

    # 定义初始状态，设置 H 储备
    initial_state = RxnState(graphs=(G_react_nx,), stage=smiles_react, slab=slab)
    final_state = RxnState(graphs=(G_prod_nx,), stage=smiles_prod, slab=slab)

    # E_salb =-418.8836059
    initial_energy = sum(atoms.get_potential_energy() for atoms in initial_state.stable_structures)
    final_energy = sum(atoms.get_potential_energy() for atoms in final_state.stable_structures)
    detal_step = (final_energy-initial_energy)/8
    # === 5. 执行 BFS 路径搜索 ===
    logger.info("Executing BFS search...")
    result_nodes = bfs_search(initial_state, G_prod_nx, n_hydrogen=8, max_paths=100, max_depth=10)

    if not result_nodes:
        logger.error("No reaction path found.")
        return

    logger.info(f"[OK] Found {len(result_nodes)} unique end nodes.")

    # === 6. 路径收集与保存 ===
    # 提取完整路径（从起始到终点）
    all_paths = collect_paths_from_nodes(result_nodes)
    unique_paths = list(set(tuple(p) for p in all_paths))
    unique_paths = [list(p) for p in unique_paths]
    logger.info(f"[OK] Found really unique {len(unique_paths)} reaction paths.")
    # 保存第一个找到的路径为轨迹文件 (extxyz)
    # result_nodes[0].save_reaction_path("path_result_Cu.extxyz")
    # result_nodes[2].save_reaction_path("path_result_Cu_1.extxyz")
    # result_nodes[8].save_reaction_path("path_result_Cu_2.extxyz")
    # 序列化搜索结果
    save_search_results(result_nodes)

    # === 7. 可视化输出 ===
    logger.info("Generating visualization files...")

    # 绘制反应树图
    plot_reaction_tree(all_paths)
    draw_reaction_networks(all_paths)
    # 导出路径中每一步的 2D 分子结构图
    output_dir = "rpfflow/tests/visuals"
    os.makedirs(output_dir, exist_ok=True)

    for i, state_smiles in enumerate(all_paths[0]):
        m = create_mol(state_smiles, add_h=True)
        save_molecule_2d(m, os.path.join(output_dir, f"step_{i}_{state_smiles}.png"))

    logger.info("Regression test completed successfully.")
    print("Done")


if __name__ == "__main__":
    main()
    loaded_paths = load_search_results()
    # from rpfflow.utils.process import replace_slab
    # slab = read("rpfflow/tests/Ag.xyz")
    # pt_path = replace_slab(loaded_paths[0], slab)
    # pt_path.save_reaction_path("path_result_Ag.extxyz")
    #
    # slab = read("rpfflow/tests/Pt.xyz")
    # pt_path = replace_slab(loaded_paths[0], slab)
    # pt_path.save_reaction_path("path_result_Pt.extxyz")
    # path_test = [["A", "B", "C", "D"],
    #              ["A", "B", "E", "D"],
    #              ["A", "B", "C", "F"],
    #              ["A", "E", "C", "D", "G"],
    #              ]
    # plot_reaction_tree(path_test, file_name="test_reaction_tree.png")
    # print("main.py end.")



