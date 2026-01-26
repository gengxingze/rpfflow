import networkx as nx
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from ase import Atoms

# =========================
# 1. Chemical system state
# =========================

@dataclass
class GraphState:
    """
    描述一个化学体系状态：
    - graph: 若干个分子/吸附物/表面片段 (每个是 nx.Graph)
    - hydrogen: 当前体系可用氢原子数（或H*数）
    - meta: 搜索与反应阶段控制信息
    """

    graph: List[nx.Graph] = field(default_factory=list)
    hydrogen: int = 0
    meta: Dict[str, Any] = field(default_factory=lambda: {

        # ===== 反应阶段 =====
        "stage": "adsorption",
        # adsorption / reduction / coupling / desorption / solution

        # ===== 组成信息 =====
        "n_carbon": 0,
        # 含有C的图在graph中的索引
        "C_graph_index": [],
        "has_CC": False,

        # ===== 表面相关 =====
        "has_surface": False,
        "desorption_count": 0,

        # ===== 结构信息 =====
        "fragments": 0,   # 连通分量个数（= len(graph)）

        # ===== 搜索控制 =====
        "allow_coupling": False,
        "allow_desorption": False,
        "adsorption_site": "all",

        # ===== 搜索评估 =====
        "penalty": 0.0
    })

    # ---------- 自动维护信息 ----------
    def update(self):
        """
        根据当前 graph / hydrogen 自动更新 meta 信息
        在每次图发生变化后调用
        """
        graphs = self.graph
        self.hydrogen = 0
        # =========================
        # 1. 基本结构信息
        # =========================
        self.meta["fragments"] = len(graphs)

        # =========================
        # 2. 含碳信息
        # =========================
        C_indices = []
        n_carbon = 0
        has_CC = False
        desorption_count = 0

        for i, g in enumerate(graphs):
            symbols = nx.get_node_attributes(g, "symbol")
            carbon_nodes = [n for n, s in symbols.items() if s == "C"]
            desorption_count = desorption_count + len([n for n, s in symbols.items() if s == "f"])
            self.hydrogen = self.hydrogen + len([n for n, s in symbols.items() if s == "H"])
            if carbon_nodes:
                C_indices.append(i)
                n_carbon += len(carbon_nodes)

                # 是否存在 C-C 键
                for u, v in g.edges():
                    if symbols[u] == "C" and symbols[v] == "C":
                        has_CC = True

        self.meta["C_graph_index"] = list(set(C_indices))

        for i in self.meta["C_graph_index"]:
            symbols = nx.get_node_attributes(graphs[i], "symbol")
            desorption_count = desorption_count + len([n for n, s in symbols.items() if s == "F"])

        self.meta["n_carbon"] = n_carbon
        self.meta["has_CC"] = has_CC
        self.meta["desorption_count"] = desorption_count

        # =========================
        # 3. 表面相关（F 作为 surface）
        # =========================
        # has_surface = False
        # for g in graphs:
        #     if any(data.get("symbol") == "F" for _, data in g.nodes(data=True)):
        #         has_surface = True
        #         break
        #
        # self.meta["has_surface"] = has_surface

    # ---------- 工具函数 ----------

    def copy(self) -> "GraphState":
        """深拷贝一个新状态（用于搜索展开）"""
        new_graph = [g.copy() for g in self.graph]
        new_meta = dict(self.meta)
        return GraphState(new_graph, self.hydrogen, new_meta)

    def summary(self) -> str:
        """调试/打印用"""
        return (f"C={self.meta['n_carbon']} | "
                f"frag={self.meta['fragments']} | "
                f"CC={self.meta['has_CC']} | "
                f"H={self.hydrogen} | "
                f"stage={self.meta['stage']}")

    def __repr__(self):
        return f"<GraphState {self.summary()}>"

    # 找到每个self.graph中每个与原子"F"相连得原子确定为吸附原子，并添加一个节点得["create"]标记为True
    # def create_ads_mark(self):
    #     for i, g in enumerate(self.graph):
    #         # 找到所有 F 节点
    #         f_nodes = [n for n, d in g.nodes(data=True) if d.get("symbol") == "F"][-1]
    #         f_neighbors = g.neighbors(f_nodes)
    #         self.graph[i].nodes[f_neighbors]["create"] = True
    #         self.graph[i].remove_node(f_nodes)
    #
    # def generate_ase_structure(self):
    #     self.create_ads_mark()
    #     self.ase_structure = []
    #     for i, g in enumerate(self.graph):
    #         self.ase_structure.append(nx_to_ase(g))

# =========================
# 2. Search tree node
# =========================

@dataclass
class SearchNode:
    """
    搜索树节点：
    - state: 当前体系状态 (GraphState)
    - node_id: 节点唯一编号
    - parent: 父节点
    - action: 从父节点到该节点的化学动作（断键/成键/加氢等）
    - hydrogen_cost: 累积氢消耗 / 电子代价
    """

    state: GraphState
    node_id: int
    parent: Optional["SearchNode"] = None
    action: Optional[str] = None
    hydrogen_cost: float = 0.0
    depth: int = field(init=False)

    def __post_init__(self):
        self.depth = 0 if self.parent is None else self.parent.depth + 1

    def path(self) -> List["SearchNode"]:
        """回溯节点路径"""
        node, nodes = self, []
        while node:
            nodes.append(node)
            node = node.parent
        return nodes[::-1]

    def states(self) -> List[GraphState]:
        """回溯体系状态路径"""
        return [n.state for n in self.path()]


    def reaction_path(self):
        path = self.path()
        steps = []
        for i in range(1, len(path)):
            steps.append({
                "step": i,
                "action": path[i].action,
                "state": path[i].state,
                "H_cost": path[i].hydrogen_cost - path[i-1].hydrogen_cost
            })
        return steps

    def __repr__(self):
        return (f"<Node id={self.node_id} depth={self.depth} "
                f"action={self.action} H_cost={self.hydrogen_cost:.2f} "
                f"{self.state.summary()}>")


@dataclass
class StructureItem:
    graph: object
    ase: Atoms
    ads_ase: Atoms = None     # 替换F后的结构
    aligned: bool = False    # 是否已做空间规范化
    meta: dict = None        # 能量/标签/位点等






