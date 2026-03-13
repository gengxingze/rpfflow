import logging
from itertools import combinations
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Iterable, Tuple
from rpfflow.core.state import RxnState
from rpfflow.core.structure import create_common_molecules
from rpfflow.rules.basica import dissociate, associate
from rpfflow.rules.matchs import is_isomorphic, is_duplicate
from rpfflow.utils.graph_ops import split_graph, merge_graphs

logger = logging.getLogger(__name__)

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


class AssociationAction(ReactionAction):
    def apply(self, state: RxnState) -> Iterable[Tuple[RxnState, str, float]]:
        for idx in state.carbon_indices:
            graph = state.graphs[idx]
            # 找所有有空价键的 C/N
            active_nodes = [
                n for n, d in graph.nodes(data=True)
                if d.get("symbol") in {"C", "N"} and d.get("valence", 0) > 0
            ]

            if len(active_nodes) < 2:
                continue

            # 任意两两组合
            for u, v in combinations(active_nodes, 2):
                new_graph = deepcopy(graph)

                # 尝试成键
                bonded = associate(new_graph, u, v, bond_order=1.0)
                if bonded is None:
                    continue

                # 更新状态
                other_graphs = [g for i, g in enumerate(state.graphs) if i != idx]
                new_graphs = other_graphs + [bonded]
                if state:
                    logger.error("Here  state is None")
                yield state.derive(new_graphs=new_graphs, h_cost=0),f"associate", 0


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
                            # print("split F")
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


# 酸性条件+H?
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
                            other_graphs = [g for i, g in enumerate(state.graphs) if i != idx]
                            new_graphs = [bonded_graph] + other_graphs
                            yield state.derive(new_graphs=new_graphs, h_cost=1), f"Add H at {n}", 1.0
                            # print(f"Add H at {n}")
                        else:
                            logger.warning("add_hydrogen error 1 !")
                    else:
                        logger.warning("add_hydrogen error 2 !")


class CouplingAction(ReactionAction):
    def apply(self, state: RxnState) -> Iterable[Tuple[RxnState, str, float]]:
        # 前置条件判断：有超过两个C/N片段，尝试不同耦合
        indices = state.element_indices({"C", "N"})
        if len(indices) >= 2:
            # 任意两个片段组合
            for idx1, idx2 in combinations(indices, 2):

                g1 = deepcopy(state.graphs[idx1])
                g2 = deepcopy(state.graphs[idx2])

                # 找可偶联原子
                nodes1 = [n for n, d in g1.nodes(data=True) if d.get("symbol") in {"C", "N"} and d.get("valence", 0) > 0]
                nodes2 = [n for n, d in g2.nodes(data=True) if d.get("symbol") in {"C", "N"} and d.get("valence", 0) > 0]
                f1_node = [n for n, d in g1.nodes(data=True) if d.get("symbol") in {"F"}][0]
                f2_node = [n for n, d in g2.nodes(data=True) if d.get("symbol") in {"F"}][0]
                if not f1_node or not f2_node:
                    continue
                # 两条链都有空的价态
                if bool(nodes1) and bool(nodes2):
                    # 找到标记F
                    g1.nodes[f1_node]["create"] = True
                    g2.nodes[f2_node]["create"] = True

                # 一个链有空， 一个链没有空
                if (bool(nodes1) and not bool(nodes2)) or (bool(nodes2) and not bool(nodes1)):
                    # 对于没有空的链，断了F，释放出空位
                    if not bool(nodes1):
                        nodes1 = list(g1.neighbors(f1_node))
                        g1.nodes[f1_node]["create"] = True
                    else:
                        nodes2 = list(g2.neighbors(f2_node))
                        g2.nodes[f2_node]["create"] = True


                # 尝试所有配对
                for n in nodes1:
                    for m in nodes2:
                        g1_copy = deepcopy(g1)
                        g2_copy = deepcopy(g2)
                        g1_copy.nodes[n]["create"] = True
                        g2_copy.nodes[m]["create"] = True
                        frag = self._couple_fragments(g1_copy, g2_copy)
                        f_node = [n for n, d in frag.nodes(data=True) if d.get("symbol") in {"F"} and d.get("create", False)]
                        for ff in f_node:
                            frag_copy = deepcopy(frag)
                            n_f = list(frag_copy.neighbors(ff))[0]
                            frag_copy = dissociate(frag_copy, n_f, ff)

                            frag_copy = self._clean(frag_copy)
                            frag_copy = split_graph(frag_copy)
                            # yield 产出：新状态、动作描述、这一步的氢消耗
                            logger.info(f"{state}")
                            yield state.derive(new_graphs=frag_copy, h_cost=0), f"couplingAction", 0

                # 两条链都没有空， 但是只有总的碳链只有两
                if (not bool(nodes1)) and (not bool(nodes2)) and (len(indices) == 2) and state.h_reserve <= 2 :
                    g1_copy = deepcopy(g1)
                    g2_copy = deepcopy(g2)
                    # 找到两条链中与F相连的{N,F}的节点号
                    nodes1 = list(g1.neighbors(f1_node))
                    g1_copy.nodes[nodes1[0]]["create"] = True
                    g1_copy = dissociate(g1_copy, f1_node, nodes1[0])

                    nodes2 = list(g2.neighbors(f2_node))
                    g2_copy = dissociate(g2_copy, f2_node, nodes2[0])
                    g2_copy.nodes[nodes2[0]]["create"] = True

                    frag = self._couple_fragments(g1_copy, g2_copy)
                    frag = self._clean(frag)
                    logger.info(f"{state}")
                    frag = split_graph(frag)
                    yield state.derive(new_graphs=frag, h_cost=0), f"couplingAction", 0


    @staticmethod
    def _clean(graph):
        """移除 create 标记"""
        for node in graph.nodes:
            graph.nodes[node].pop("create", None)
        return graph

    @staticmethod
    def _couple_fragments(g1, g2):
        merged = merge_graphs([g1, g2])
        # 找到合并后的节点
        couple_nodes = [node for node, d in merged.nodes(data=True) if d.get("create",False) and d.get("symbol") in {"C", "N"}]
        coupled = associate( merged, couple_nodes[0],couple_nodes[1], bond_order=1.0, enforce=True)
        return coupled


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
    from rpfflow.rules.basica import update_valence
    from rpfflow.utils.visualizer import plot_molecular_graph, save_molecule_2d
    # === 构建反应物 / 生成物 ===
    mol_react = create_mol('[C](F)(O)O', add_h=True)                 # CO2 (或简化占位)
    mol_prod  = create_mol("C", add_h=True)     # CH3OH

    G_react = rdkit_to_nx(mol_react)
    G_prod  = rdkit_to_nx(mol_prod)

    update_valence(G_react)
    update_valence(G_prod)


    from ase.io import read
    from rpfflow.core.structure import get_reference_structure, create_mol
    slab = read("../../tests/POSCAR")
    G = RxnState(graphs=(G_react,G_react), h_reserve=16, stage="[O]C(=O)F", reference_structure=get_reference_structure(slab))
    action = CouplingAction()

    results = list(action.apply(G))

    plot_molecular_graph(G.graphs[0])

    results_2 = list(AssociationAction().apply(G))
    results_3 = list(HydrogenationAction().apply(G))
    print(f"生成状态数: {len(results_2)}")



