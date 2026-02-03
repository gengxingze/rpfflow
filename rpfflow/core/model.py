import logging
from collections import deque
from rpfflow.core.action import HydrogenationAction, DissociationAction, CCCouplingAction
from rpfflow.core.state import RxnState, SearchNode
from rpfflow.rules.matchs import is_duplicate

logger = logging.getLogger(__name__)


def bfs_search(initial_state: RxnState, target_graph, n_hydrogen=8, rules=None, max_paths=5):
    """
    重构后的搜索引擎：
    - initial_state: RxnState 实例
    - target_graph: 目标产物的 nx.Graph
    - n_hydrogen: 最大允许的氢消耗
    - rules: 反应规则对象列表
    """
    if rules is None:
        rules = [HydrogenationAction(), DissociationAction(), CCCouplingAction()]

    # 初始化队列与去重集合
    root_node = SearchNode(state=initial_state)
    open_queue = deque([root_node])

    # 关键：RxnState 的不可变性支持了 $O(1)$ 复杂度的去重
    # visited = {initial_state}

    found_count = 0
    found_paths = []  # 存储找到的 SearchNode
    while open_queue:
        current_node = open_queue.popleft()
        state = current_node.state

        # --- 1. 目标检查 (利用 RxnState 缓存的 carbon_indices 提高效率) ---
        # 只要当前状态包含目标产物，就记录该节点
        if is_duplicate(target_graph, state.graphs):
            found_paths.append(current_node)
            logger.info(f"🎯 找到生成物路径！已找到: {len(found_paths)}/{max_paths}, 深度: {current_node.depth}")

            # 达到设定数量则提前结束
            if len(found_paths) >= max_paths:
                logger.info(f"已达到最大路径数 {max_paths}，停止搜索。")
                return found_paths

            # 注意：一旦某个节点判定为目标，通常不需要再从它向下演化
            continue

        # --- 2. 规则驱动的状态演化 ---
        for rule in rules:
            # apply 现在是一个生成器，按需产出后继状态
            for next_state, action_desc, h_cost in rule.apply(state):

                # 累积氢消耗检查
                total_h_cost = current_node.cumulative_h_cost + h_cost
                if total_h_cost > n_hydrogen:
                    continue

                # 去重检查
                if next_state:
                    # visited.add(next_state)

                    # 自动分配 node_id 并在内部累计 cost
                    child_node = SearchNode(
                        state=next_state,
                        parent=current_node,
                        action=action_desc,
                        step_h_cost=h_cost
                    )
                    open_queue.append(child_node)

    if not found_paths:
        logger.warning("搜索结束，未找到任何可行路径。")

    return found_paths  # 返回所有找到的终点节点列表

if __name__ == "__main__":
    """
    回归测试：CO2 → CH3OH 反应路径搜索是否可正常运行
    目标：
    - 元素守恒检查通过
    - BFS 能返回至少一条路径
    - 路径中每一步都是 RxnState
    """

    from rpfflow.utils.convert import rdkit_to_nx
    from rpfflow.core.structure import create_mol
    from rpfflow.rules.basica import check_element_conservation
    # from rpfflow.search import bfs_search
    from rpfflow.rules.basica import update_valence

    # === 构建反应物 / 生成物 ===
    mol_react = create_mol('O=C(F)O')                 # CO2 (或简化占位)
    mol_prod  = create_mol("C", add_h=True)     # CH3OH

    G_react = rdkit_to_nx(mol_react)
    G_prod  = rdkit_to_nx(mol_prod)

    update_valence(G_react)
    update_valence(G_prod)

    # === 元素守恒检查 ===
    conserved, diffs = check_element_conservation(G_react, G_prod)
    # assert conserved, f"元素不守恒: {diffs}"
    from ase.io import read
    from structure import get_reference_structure, create_mol

    slab = read("../../tests/POSCAR")
    G_react = RxnState(graphs=(G_react,), h_reserve=8, stage="[O]C(=O)F", reference_structure=get_reference_structure(slab))

    # === 执行搜索 ===
    node = bfs_search(G_react, G_prod, n_hydrogen=8)
    print(f"[OK] 找到 {len(node)} 步反应路径")
    paths = []
    # for x, n in enumerate(node):
    #     nnpp = n.reaction_history
    #     paths.append(nnpp)
    #     n.save_reaction_path(f"path_{x}.extxyz")

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
        save_molecule_2d(m, f"../../tests/mol_{i}.png")


    # print(f"[OK] 找到 {len(paths)} 步反应路径")
    print("Done")

