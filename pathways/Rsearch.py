import networkx as nx
from copy import deepcopy
from collections import deque
from rules.matchs import is_isomorphic, is_duplicate, match_target
from rules.basica import dissociate, associate, update_valence
from graph_mm.graph_ops import split_graph, merge_graphs
from graph_mm.molgraph import create_common_molecules

import logging

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,                 # 日志级别: DEBUG < INFO < WARNING < ERROR < CRITICAL
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# 创建一个 logger
logger = logging.getLogger(__name__)

def bfs_search(G_start, G_target, n_hydrogen=10):
    """
    使用 BFS 搜索反应路径：从反应物 G_start 到生成物 G_target
    """
    queue = deque([(G_start, [], n_hydrogen)])  # (当前图, 操作序列, 剩余氢数)
    reaction_path = []
    # 常见小分子
    molecules = create_common_molecules()
    OH_ = molecules["OH_"]
    H = molecules["H"]
    H2O = molecules["H2O"]
    R = molecules["F"]
    OCOR = molecules["OCOR"]
    OCRO = molecules["OCRO"]
    nn = 0
    # 初始阶段只允许添加催化点位一次
    add_catalytic_site = True
    while queue:
        current_graph, actions, hydrogen = queue.popleft()
        print(actions)
        nn = nn+1
        # --- 检查是否匹配目标 ---
        subgraphs = split_graph(current_graph)
        if is_duplicate(G_target, subgraphs):
            print(f"🎯 找到生成物！反应路径： {nn}")
            reaction_path.append(actions)

        # === 若氢数不足则跳过该路径 ===
        if hydrogen < 0:
            # raise ValueError("HYDROGEN <UNK>")
            continue

        # === 初始化时添加催化基团 ===
        if add_catalytic_site:
            queue.append((OCRO, actions + [OCRO], hydrogen))
            queue.append((OCOR, actions + [OCOR], hydrogen))
            add_catalytic_site = False
            continue

        # =====================================================
        # Case 1: 所有原子价态饱和 → 尝试断键
        # =====================================================
        if all(current_graph.nodes[n]["valence"] <= 0 for n in current_graph.nodes):
            # --- 生成候选图 ---
            for u, v in list(current_graph.edges()):
                cut_graph = dissociate(deepcopy(current_graph), u, v)
                fragments = split_graph(cut_graph)

                # 检查键的断开是否使图变成互不相同的子图
                # --- 单图情况 ---
                if len(fragments) == 1:
                    update_valence(cut_graph)
                    queue.append((cut_graph, actions + [cut_graph], hydrogen))

                # --- 双图情况 ---
                if len(fragments) == 2:
                    # 检查生成的子图是否有HO-,有则消耗一个氢使H0-变成H20, G_cut 变成非H20的图
                    if is_duplicate(OH_, fragments):
                        hydrogen = hydrogen - 1
                        non_oh =  [g for g in fragments if not is_isomorphic(OH_, g)][0]
                        update_valence(non_oh)
                        queue.append((non_oh, actions + [non_oh], hydrogen))
                    # 如果H2O在G_target中；
                    if is_duplicate(H2O, fragments):
                        non_h2o = [g for g in fragments if not is_isomorphic(OH_, g)][0]
                        update_valence(non_h2o)
                        queue.append((non_h2o, actions + [non_h2o], hydrogen))
                    # 情况3: 若氢量较低且涉及催化位点，则允许催化断裂
                    if hydrogen < 4 and (current_graph.nodes[u]["symbol"] == "F" or current_graph.nodes[v]["symbol"] == "F"):
                        G_middle = [g for g in fragments if not is_isomorphic(R, g)][0]
                        update_valence(G_middle)
                        queue.append((G_middle, actions + [G_middle], hydrogen))

        # =====================================================
        # Case 2: 存在未饱和原子 → 尝试加氢
        # =====================================================
        else:
            for n in current_graph.nodes:
                if current_graph.nodes[n]["valence"] > 0:
                    candidate_graph = deepcopy(current_graph)
                    candidate_graph.nodes[n]["create"] = True
                    H.nodes[0]["create"] = True
                    add_graph = merge_graphs([candidate_graph, H])
                    # 找到新加入的氢节点
                    id_nodes = [n for n, d in add_graph.nodes(data=True) if d.get("create", False)]
                    if len(id_nodes) != 2:
                        logger.error("新氢添加错误：未识别新节点")
                        continue
                    bonded_graph = associate(add_graph, id_nodes[0], id_nodes[1], bond_order= 1.0)

                    # 清除标志位
                    bonded_graph.nodes[id_nodes[0]]["create"] = False
                    bonded_graph.nodes[id_nodes[1]]["create"] = False

                    if bonded_graph is not None:
                        hydrogen = hydrogen - 1
                        update_valence(deepcopy(bonded_graph))
                        queue.append(
                            (bonded_graph, actions + [bonded_graph], hydrogen))


    return reaction_path


if __name__ == "__main__":
    from graph_mm.molgraph import create_mol, rdkit_to_nx, create_common_molecules, nx_to_rdkit, rdkit_ase
    from graph_mm.visualizer import draw_graph, draw_graph_list

    # === 反应物：CO2 ===
    mol_react = create_mol("O=C=O")
    G_react = rdkit_to_nx(mol_react)
    update_valence(G_react)
    # === 生成物：CH3OH ===
    mol_prod = create_mol("C", add_h=True)
    G_prod = rdkit_to_nx(mol_prod)
    update_valence(G_prod)

    from graph_mm.graph_ops import merge_graphs

    # visualize_graph(G_react, "CO2", save_path="CO2")
    # visualize_graph(G_prod, "CH3OH", save_path="CH3OH")

    from rules.basica import check_element_conservation
    conserved, diffs = check_element_conservation(G_react, G_prod)
    print("元素是否守恒:", conserved)
    if not conserved:
        print("不守恒的元素:", diffs)

    # === 运行搜索 ===
    path = bfs_search(G_react, G_prod)
    print(f"一共找到 {len(path)} 条可能路径)")
    # for pp in path:
    #     draw_graph_list(pp)
    # visualize_graph(path[0])

    from graph_mm.molgraph import rdkit_ase
    from mace.calculators import mace_off

    calc = mace_off(model="medium", device='cpu')
    for pp in path[0]:
        atoms = rdkit_ase(nx_to_rdkit(pp))
        atoms.calc = calc
        print(atoms.get_potential_energy())



    from rdkit import Chem
    from rdkit.Chem import Draw

    # 假设 mol_list 是 Chem.Mol 对象的列表
    # mol_list = [mol1, mol2, mol3, ...]
    # mol_list = []
    # for mm in path:
    #     mol_list.append(nx_to_rdkit(mm))
    # # 画在同一张图中，网格显示
    # img = Draw.MolsToGridImage(
    #     mol_list,
    #     molsPerRow=len(mol_list),  # 每行显示多少个分子，这里全部一行
    #     subImgSize=(200, 200),  # 每个子图大小
    #     legends=[Chem.MolToSmiles(mol) for mol in mol_list] # 可选：每个分子下的文字
    # )
    #
    # # 展示图片（Jupyter Notebook中可用）
    # img.show()
    print("successful")
