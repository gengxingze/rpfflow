import networkx as nx
from copy import deepcopy
from collections import deque
from rules.matchs import is_isomorphic, is_duplicate, match_target
from rules.basica import dissociate, associate, update_valence
from graph_mm.graph_ops import split_graph, merge_graphs
from graph_mm.molgraph import create_common_molecules, rdkit_to_ase, nx_to_rdkit
from pathways.gp_data import GraphState, SearchNode

import logging

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,                 # 日志级别: DEBUG < INFO < WARNING < ERROR < CRITICAL
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# 创建一个 logger
logger = logging.getLogger(__name__)

def bfs_search(state_start, state_target, n_hydrogen=8):
    """
    使用 BFS 搜索反应路径：从反应物 state_start 到生成物 state_target
    """

    # 常见小分子
    molecules = create_common_molecules()
    OH_ = molecules["OH_"]
    H = molecules["H"]
    H2O = molecules["H2O"]
    R = molecules["F"]
    OCRO = molecules["OCRO"]

    graph_state = GraphState(graph=[OCRO], meta={"n_carbon": 1, "fragment": 1})
    graph_state.update()
    node_id = 0
    root_node = SearchNode(state=graph_state, node_id=node_id, parent=None, action="start", hydrogen_cost=0.0)
    open_queue = deque([root_node])

    xxx = 1

    while open_queue:
        current_node = open_queue.popleft()
        current_state = current_node.state

        # --- 检查是否匹配目标 ---
        subgraphs = split_graph(current_state.graph[current_state.meta["C_graph_index"][-1]])

        if is_duplicate(state_target, subgraphs):
            xxx = xxx + 1
            print(f"🎯 找到生成物！反应路径： {xxx}")
            return current_node

        # === 若氢数不足则跳过该路径 ===
        if current_state.hydrogen > n_hydrogen:
            # raise ValueError("HYDROGEN <UNK>")
            continue

        reactions = []
        # =====================================================
        # Case 1: 所有原子价态饱和 → 尝试断键
        # =====================================================
        for idx in current_state.meta["C_graph_index"]:
            current_graph = deepcopy(current_state.graph[idx])
            if all(current_graph.nodes[n]["valence"] <= 0 for n in current_graph.nodes):
                # --- 生成候选图 ---
                for u, v in list(current_graph.edges()):
                    cut_graph = dissociate(deepcopy(current_graph), u, v)

                    # 对于双碳检查是否是C-C键，禁止断开成型的C-C。 ！！现有逻辑不会断裂C-C
                    # if len(current_state.meta["C_graph_index"]) == 2:
                    #     pass
                    fragments = split_graph(cut_graph)

                    # 检查键的断开是否使图变成互不相同的子图
                    # --- 单图情况 ---
                    if len(fragments) == 1:
                        middle_state = GraphState(graph=fragments)
                        reactions.append([middle_state, f"dissociate: {u, v}", idx, 0])

                    # --- 双图情况 ---
                    if len(fragments) == 2:
                        # 检查生成的子图是否有HO-,有则消耗一个氢使H0-变成H20, G_cut 变成非H20的图
                        if is_duplicate(OH_, fragments):
                            fragments = [g for g in fragments if not is_isomorphic(OH_, g)] + [H2O]
                            middle_state = GraphState(graph=fragments)
                            reactions.append([middle_state, f"dissociate: {u, v}", idx, 1])

                        # 如果H2O在G_target中；
                        if is_duplicate(H2O, fragments):
                            middle_state = GraphState(graph=fragments)
                            reactions.append([middle_state, f"dissociate: {u, v}", idx, 0])

                        # 情况3: 若氢量较低且涉及催化位点，则允许催化断裂
                        print("H=", n_hydrogen - current_node.hydrogen_cost)
                        if (n_hydrogen - current_node.hydrogen_cost) < 2 and (
                                current_graph.nodes[u]["symbol"] == "F" or current_graph.nodes[v]["symbol"] == "F"):
                            middle_state = GraphState(graph=fragments)
                            reactions.append([middle_state, f"dissociate: {u, v}", idx, 0])

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
                        bonded_graph = associate(add_graph, id_nodes[0], id_nodes[1], bond_order=1.0)

                        # 清除标志位
                        bonded_graph.nodes[id_nodes[0]]["create"] = False
                        bonded_graph.nodes[id_nodes[1]]["create"] = False

                        if bonded_graph is not None:
                            middle_state = GraphState(graph=[bonded_graph])
                            reactions.append([middle_state, f"Add H at {n}", idx, 1])


                        # 对于双碳检查是否有C-C键，如果没有，且其中1个碳未饱和键则尝试，与另一个C构成C-C键。
                        print(current_state.meta["desorption_count"])
                        if (not current_state.meta["has_CC"]) and (current_graph.nodes[n]["symbol"] == "C") and (
                                len(current_state.meta["C_graph_index"]) == 2) and (current_state.meta["desorption_count"] == 2):
                            # 确定另一个碳链的C是否
                            other_graph = current_state.graph[current_state.meta["C_graph_index"][1 - idx]]
                            # 找另一个图中的碳原子
                            m = [m for m in other_graph.nodes
                                             if other_graph.nodes[m]["symbol"] == "C"][0]
                            # 找另一个图中的R原子
                            f = [f for f in other_graph.nodes
                                 if other_graph.nodes[f]["symbol"] == "F"][0]

                            g1 = deepcopy(current_graph)
                            g2 = deepcopy(other_graph)

                            # 如果另一个C未饱和，先去掉吸附R然后构成C-C
                            # 如果另一个C饱和，先判断是否有C-R键，如果有则断掉R形成C-C
                            if (other_graph.nodes[m]["valence"] > 0) or other_graph.has_edge(m, f):
                                # 打 create 标记用于 merge 后定位原子
                                g2.nodes[m]["create"] = True
                                g2.nodes[f]["create"] = True

                                merged = merge_graphs([g1, g2])
                                id_nodes = [mm for mm in merged.nodes
                                     if merged.nodes[mm]["symbol"] == "C"]
                                new_graph = associate(merged, id_nodes[0], id_nodes[1], bond_order=1.0)

                                if new_graph is not None:
                                    id_nodes = [n for n, d in new_graph.nodes(data=True) if d.get("create", False)]
                                    # 断掉与与F相连的边，催化剂脱附
                                    f = [
                                        n for n, d in new_graph.nodes(data=True)
                                        if d.get("create", False) and d.get("symbol") == "F"
                                    ][0]
                                    ff = list(new_graph.neighbors(f))[0]

                                    cut_graph = dissociate(new_graph, f, ff)
                                    cut_graph.nodes[id_nodes[0]]["create"] = False
                                    cut_graph.nodes[id_nodes[1]]["create"] = False
                                    fragments = split_graph(cut_graph)
                                    new_state = GraphState(graph=fragments)
                                    new_state.update()
                                    child = SearchNode(
                                        state=new_state,
                                        node_id=node_id+1,
                                        parent=current_node,
                                        action=f"Add C-C at {n}",
                                        hydrogen_cost=current_node.hydrogen_cost + 0)
                                    open_queue.append(child)

        # =====================================================
        # 🌱 生成子节点
        # =====================================================
        for new_state, action, changed_idx, hydrogen_cost in reactions:
            node_id += 1
            # 如果原体系有两个图，把“没变的那个”补回来
            if len(current_state.meta["C_graph_index"]) == 2:
                other_graph = current_state.graph[current_state.meta["C_graph_index"][1 - changed_idx]]
                new_state.graph = new_state.graph + [deepcopy(other_graph)]
            new_state.update()
            child = SearchNode(
            state=new_state,
            node_id=node_id,
            parent=current_node,
            action=action,
            hydrogen_cost=current_node.hydrogen_cost+hydrogen_cost)

            open_queue.append(child)


    return None


if __name__ == "__main__":
    from graph_mm.molgraph import create_mol, rdkit_to_nx, create_common_molecules, nx_to_rdkit, rdkit_to_ase
    from graph_mm.visualizer import plot_molecular_graph, save_molecule_2d, plot_molecular_graphs

    # === 反应物：CO2 ===
    mol_react = create_mol("C")
    G_react = rdkit_to_nx(mol_react)
    update_valence(G_react)
    # === 生成物：CH3OH ===
    mol_prod = create_mol("C", add_h=True)
    G_prod = rdkit_to_nx(mol_prod)
    update_valence(G_prod)

    from rules.basica import check_element_conservation
    conserved, diffs = check_element_conservation(G_react, G_prod)
    print("元素是否守恒:", conserved)
    if not conserved:
        print("不守恒的元素:", diffs)

    # === 运行搜索 ===
    path = bfs_search([G_react, G_react], G_prod, n_hydrogen=8)
    path = path.reaction_path()
    a = []
    for r in path:
        a.append(r["state"].graph[0])
    plot_molecular_graphs(a)
    print(f"一共找到 {len(path)} 条可能路径)")

    print("successful")
