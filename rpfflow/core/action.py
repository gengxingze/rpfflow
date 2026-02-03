from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Iterable, Tuple
from rpfflow.core.state import RxnState
from rpfflow.core.structure import create_common_molecules
from rpfflow.rules.basica import dissociate, associate
from rpfflow.rules.matchs import is_isomorphic, is_duplicate
from rpfflow.utils.graph_ops import split_graph, merge_graphs


molecules = create_common_molecules()
OH_ = molecules["OH_"]
H = molecules["H"]
H2O = molecules["H2O"]
R = molecules["F"]



class ReactionAction(ABC):
    """化学反应动作基类"""
    @abstractmethod
    def apply(self, state: RxnState) -> Iterable[Tuple[RxnState, str, float]]:
        """
        返回: Iterator[(新状态, 动作描述, 氢消耗)]
        """
        pass


class DissociationAction(ReactionAction):
    def apply(self, state: RxnState) -> Iterable[Tuple[RxnState, str, float]]:
        # 获取预定义的分子模型（建议从配置或外部传入）
        # OH_molecule, H2O_molecule = ...

        for idx in state.carbon_indices:
            graph = state.graphs[idx]

            # 只有当该图所有原子价态饱和时，才尝试断键
            if all(graph.nodes[n].get("valence", 0) <= 0 for n in graph.nodes):

                for u, v in list(graph.edges()):
                    # 核心逻辑保护：如果是 C-C 键，通常不在此处断裂（可根据需要开启）
                    if graph.nodes[u]["symbol"] == "C" and graph.nodes[v]["symbol"] == "C":
                        continue

                    # 执行图剪枝操作，返回碎片列表
                    frags = self._cut(graph, u, v)

                    h_cost = 0.0
                    final_frags = []

                    # --- 逻辑分支 1: 产生 2 个碎片 (化学键断裂导致分子解离) ---
                    if len(frags) == 1:
                        final_frags =frags
                    elif len(frags) == 2:
                        # 特殊逻辑 A: 检查是否存在 OH- 碎片，自动加氢生成 H2O
                        if is_duplicate(OH_, frags):
                            final_frags = [g for g in frags if not is_isomorphic(OH_, g)] + [H2O]
                            h_cost = 1
                        elif is_duplicate(H2O, frags):
                            final_frags = frags

                        # 特殊逻辑 B: 如果是催化位点 F 参与的断裂（且氢量较低时允许）
                        elif state.h_reserve <= 2 and (
                                graph.nodes[u]["symbol"] == "F" or graph.nodes[v]["symbol"] == "F"):
                            print("split F")
                            final_frags = list(frags)
                            h_cost = 0.0
                    else:
                        # 暂时不可能是三个碎片
                        continue

                    # --- 组合并产出新状态 ---
                    if final_frags:
                        # 保持除了当前正在操作的 idx 之外的其他子图不变
                        other_graphs = [g for i, g in enumerate(state.graphs) if i != idx]
                        new_graphs = other_graphs + final_frags

                        # yield 产出：新状态、动作描述、这一步的氢消耗
                        yield state.derive(new_graphs=new_graphs, h_cost=h_cost), f"dissociate: {u}-{v}", h_cost


    @staticmethod
    def _cut(graph, u, v):
        """执行物理断键并返回分连通分量"""
        # 使用之前的工具函数：断键 -> 重新计算价态 -> 分裂图
        cut_g = dissociate(deepcopy(graph), u, v)
        # update_valence(cut_g) # 确保断键后原子价态已更新
        return split_graph(cut_g)



class HydrogenationAction(ReactionAction):
    def apply(self, state: RxnState) -> Iterable[Tuple[RxnState, str, float]]:
        # H_molecule 通常是单原子 H 或 H* 的图表示
        # H_atom = ...

        for idx in state.carbon_indices:
            graph = state.graphs[idx]

            for n in graph.nodes:
                # 检查原子价态是否未饱和
                if graph.nodes[n].get("valence", 0) > 0:
                    # 1. 标记当前节点与氢节点以便合并后定位
                    g_temp = deepcopy(graph)
                    h_temp = deepcopy(H)
                    g_temp.nodes[n]["create"] = True
                    h_temp.nodes[0]["create"] = True

                    # 2. 合并图并建立化学键 (associate)
                    merged = merge_graphs([g_temp, h_temp])
                    id_nodes = [node for node, d in merged.nodes(data=True) if d.get("create", False)]

                    if len(id_nodes) == 2:
                        bonded_graph = associate(merged, id_nodes[0], id_nodes[1], bond_order=1.0)
                        if bonded_graph is not None:
                            # 3. 清理标记位
                            for node in id_nodes:
                                bonded_graph.nodes[node]["create"] = False

                            # 4. 生成新状态，消耗 1 个氢
                            new_graphs = [bonded_graph]
                            yield state.derive(new_graphs=new_graphs, h_cost=1), f"Add H at {n}", 1.0
                            print(f"Add H at {n}")
                        else:
                            print("add_hydrogen error 1 !")
                    else:
                        print("add_hydrogen error 2 !")

class CCCouplingAction(ReactionAction):
    def apply(self, state: RxnState) -> Iterable[Tuple[RxnState, str, float]]:
        # 前置条件判断：尚未形成CC键、有两个含碳片段、脱附计数达到 2
        if (not state.has_cc_bond) and \
                (len(state.carbon_indices) == 2) and \
                (state.desorption_count == 2):

            for i, idx in enumerate(state.carbon_indices):
                graph = state.graphs[idx]

                for n in graph.nodes:
                    # 只有 C 原子参与偶联尝试
                    if graph.nodes[n].get("symbol") == "C" and graph.nodes[n].get("valence", 0) > 0:

                        # 定位另一个含碳片段及其 C、F 原子
                        other_idx = state.carbon_indices[1 - i]
                        other_graph = state.graphs[other_idx]

                        # 查找另一个图中的 C 和 F 节点
                        c_nodes = [m for m, d in other_graph.nodes(data=True) if d.get("symbol") == "C"]
                        f_nodes = [f for f, d in other_graph.nodes(data=True) if d.get("symbol") == "F"]

                        if not c_nodes or not f_nodes:
                            continue

                        m, f = c_nodes[0], f_nodes[0]

                        # 如果另一个 C 未饱和，或当前通过 F 吸附，则尝试偶联
                        if (other_graph.nodes[m].get("valence", 0) > 0) or other_graph.has_edge(m, f):
                            g1, g2 = deepcopy(graph), deepcopy(other_graph)

                            # 标记关键原子以便合并后 associate
                            g1.nodes[n]["create"] = True  # 当前 C
                            g2.nodes[m]["create"] = True  # 目标 C
                            g2.nodes[f]["create"] = True  # 催化位点 F

                            merged = merge_graphs([g1, g2])
                            # 找到对应的两个 C 节点建立 C-C 键
                            c_ids = [node for node, d in merged.nodes(data=True)
                                     if d.get("symbol") == "C" and d.get("create", False)]

                            if len(c_ids) == 2:
                                coupled_g = associate(merged, c_ids[0], c_ids[1], bond_order=1.0)

                                if coupled_g is not None:
                                    # 建立 C-C 后，处理 F 脱附逻辑
                                    f_node = [node for node, d in coupled_g.nodes(data=True)
                                              if d.get("symbol") == "F" and d.get("create", False)][0]

                                    # 找到 F 的邻居并断键 (脱附)
                                    f_neighbors = list(coupled_g.neighbors(f_node))
                                    if f_neighbors:
                                        final_g = dissociate(coupled_g, f_node, f_neighbors[0])

                                        # 清理标记并分裂碎片
                                        for node in final_g.nodes:
                                            final_g.nodes[node].pop("create", None)

                                        final_fragments = split_graph(final_g)
                                        yield state.derive(final_fragments), f"Add C-C at {n}", 0.0







