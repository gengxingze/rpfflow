import time
import pickle
import networkx as nx
from ase import Atoms
from functools import cached_property
from dataclasses import dataclass, field, replace
from typing import List, Optional, Iterator, Tuple
from rpfflow.utils.convert import nx_to_ase, nx_to_rdkit
from rpfflow.core.structure import rotate_F, optimize_structure, generate_adsorption_structures


@dataclass(frozen=True)
class RxnState:
    """
    rxnflow 核心状态类：
    - graphs: 体系中所有独立片段的元组 (保持顺序且不可变)
    - h_reserve: 体系当前剩余可用的 H 原子/质子数
    - stage: 反应阶段 (adsorption, reduction, etc.)
    """
    graphs: Tuple[nx.Graph, ...]
    h_reserve: int = 0
    stage: str = "adsorption"
    penalty: float = 0.0
    reference_structure: dict[str, Atoms] = None

    # =====================================================
    # 2. 自动缓存的计算属性 (替代原来的 update() 方法)
    # =====================================================

    @cached_property
    def carbon_indices(self) -> List[int]:
        """索引出含有碳原子的子图位置"""
        return [i for i, g in enumerate(self.graphs)
                if any(nx.get_node_attributes(g, "symbol").get(n) == "C" for n in g.nodes)]

    @cached_property
    def n_carbon(self) -> int:
        """体系总碳数"""
        return sum(len([n for n, s in nx.get_node_attributes(self.graphs[i], "symbol").items() if s == "C"])
                   for i in self.carbon_indices)

    @cached_property
    def has_cc_bond(self) -> bool:
        """是否存在 C-C 键"""
        for i in self.carbon_indices:
            g = self.graphs[i]
            symbols = nx.get_node_attributes(g, "symbol")
            if any(symbols[u] == "C" and symbols[v] == "C" for u, v in g.edges()):
                return True
        return False

    @cached_property
    def desorption_count(self) -> int:
        """统计特定原子 (F/f) 的数量，用于判断脱附状态"""
        count = 0
        for g in self.graphs:
            symbols = nx.get_node_attributes(g, "symbol").values()
            count += (list(symbols).count("f") + list(symbols).count("F"))
        return count

    @cached_property
    def stable_structures(self):
        """
        根据当前的 graphs 生成对应的 ASE Atoms 对象列表
        """
        stru_ase = []

        for g in self.graphs:
            stru = nx_to_ase(g)
            formula = stru.get_chemical_formula()

            # 1. 检查是否为预设的孤立小分子
            if formula in self.reference_structure:
                stru_ase.append(self.reference_structure[formula])
                continue

            # 2. 如果结构中没有 F 原子 (通常是干净表面或已脱离表面的产物)
            if 'F' not in stru.get_chemical_symbols():
                opt_stru = optimize_structure(stru)
                stru_ase.append(opt_stru)
                continue

            # 3. 如果含有 F 原子 (作为占位符或吸附指示)
            # 且长度不为 1 (排除孤立 F 原子)
            symbols = stru.get_chemical_symbols()
            if 'F' in symbols and len(stru) > 1:
                # 旋转/调整含F的吸附质构型
                stru = rotate_F(stru)
                # 找到所有 F 原子的索引
                f_indices = [atom.index for atom in stru if atom.symbol == 'F']
                stru.pop(f_indices[0])
                # 生成吸附结构
                ads_structures_ase, _ = generate_adsorption_structures(
                    adsorbate_ase=stru,
                    slab_ase=self.reference_structure["F"]
                )

                choice_list = []
                for choice_stru in ads_structures_ase:
                    # 这里的 ss 应该是吸附在 slab 上的完整体系
                    opt_stru = optimize_structure(choice_stru)
                    choice_list.append(opt_stru)
                # 找到能量最低的结构对象
                best_stru = min(choice_list, key=lambda s: s.get_potential_energy())
                stru_ase.append(best_stru)


        # 4. 在电催化台阶图中，我们通常需要的是能量列表来绘图，而不是求和
        # 如果你确实需要总能量，可以使用 sum(energies)
        return stru_ase
    # =====================================================
    # 3. 状态演化工具
    # =====================================================

    def derive(self, new_graphs: List[nx.Graph], h_cost: int = 0, **kwargs) -> "RxnState":
        """
        生成衍生状态的快捷方式 (用于搜索展开)
        example: state.derive(new_graphs=fragments, h_cost=1, stage="reduction")
        """
        updates = {
            "graphs": tuple(new_graphs),
            "h_reserve": self.h_reserve - h_cost,
            **kwargs
        }
        # replace 函数是 dataclasses 提供的克隆并更新的方法
        return replace(self, **updates)

    def __repr__(self):
        return (f"<RxnState C={self.n_carbon} | H={self.h_reserve} | "
                f"CC={self.has_cc_bond} | Stage={self.stage}>")


# =========================
# 2. Search tree node
# =========================
@dataclass(frozen=True)
class SearchNode:
    """
    rpfflow 搜索树节点：
    记录从初始状态到当前状态的演化轨迹。
    """
    state: 'RxnState'
    parent: Optional['SearchNode'] = field(default=None, repr=False)  # 避免循环打印
    action: str = "initial_state"

    # 代价管理
    step_h_cost: float = 0.0  # 当前步骤的氢消耗
    cumulative_h_cost: float = 0.0  # 累计氢消耗
    priority_score: float = 0.0  # 用于 A* 搜索的启发式分数 (g + h)

    # 自动生成的元数据
    node_id: int = field(default_factory=lambda: int(time.time() * 1e6) % 10 ** 8)
    depth: int = 0

    def __post_init__(self):
        # 自动计算深度和累计代价
        if self.parent is not None:
            # 这里的 object.__setattr__ 是因为 frozen=True
            object.__setattr__(self, 'depth', self.parent.depth + 1)
            object.__setattr__(self, 'cumulative_h_cost', self.parent.cumulative_h_cost + self.step_h_cost)

    # =====================================================
    # 路径回溯工具 (使用生成器节省内存)
    # =====================================================

    def iter_path(self) -> Iterator['SearchNode']:
        """从根节点到当前节点的迭代器"""
        path = []
        curr = self
        while curr:
            path.append(curr)
            curr = curr.parent
        yield from reversed(path)

    @property
    def reaction_history(self) -> List[dict]:
        """
        生成从根节点开始的完整反应路径描述。
        """
        history = []
        for node in self.iter_path():
            # 我们不再跳过根节点，而是根据是否为根节点来填充描述
            is_root = node.parent is None

            history.append({
                "step": node.depth,
                "action": "START" if is_root else node.action,
                "h_cost": 0.0 if is_root else node.step_h_cost,
                "total_h": node.cumulative_h_cost,
                "state": node.state
            })
        return history
    # =====================================================
    # 辅助工具
    # =====================================================
    def __repr__(self):
        return (f"Node(id={self.node_id}, depth={self.depth}, "
                f"action='{self.action}', cost={self.cumulative_h_cost}, "
                f"state={self.state})")

    def save_reaction_path(self, filename="reaction_path.extxyz"):
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.io import write
        path_frames = []
        history = self.reaction_history
        for entry in history:
            structures = entry["state"].stable_structures  # 这是一个 [Atoms, Atoms, ...] 列表

            if not structures:
                continue

            for i, atoms in enumerate(structures):
                # 1. 创建副本，防止修改原始数据
                temp_atoms = atoms.copy()

                # 2. 提取并注入能量和受力
                # 这样 write 时，每个碎片都会带有它自己的物理信息
                try:
                    energy = atoms.get_potential_energy()
                    forces = atoms.get_forces()
                    temp_atoms.calc = SinglePointCalculator(
                        temp_atoms, energy=energy, forces=forces
                    )
                except Exception as e:
                    # 如果没有计算器信息，则跳过计算器设置
                    print(f"Step {entry['step']} fragment {i} has no energy/force: {e}")

                # 3. 注入元数据
                temp_atoms.info['step'] = entry['step']
                temp_atoms.info['action'] = entry['action']
                temp_atoms.info['fragment_id'] = i  # 标记这是该步骤中的第几个碎片

                path_frames.append(temp_atoms)

        # 4. 写入文件
        write(filename, path_frames)
        print(f"成功导出 {len(path_frames)} 个独立碎片结构至 {filename}")


def get_state_signature(state: RxnState):
    """
    生成 RxnState 的唯一签名：
    - simple_smiles: 折叠氢原子后的紧凑格式，适合绘图标签 (例如 [CH4], [CH3OH])
    - complex_smiles: 全显式氢格式，包含所有拓扑连接，适合底层对比
    """
    from rdkit import Chem
    simple_list = []
    complex_list = []

    for g in state.graphs:
        # 1. 基础转换 (此时含有 nx.Graph 中定义的独立 H 节点)
        mol = nx_to_rdkit(g)

        # --- 生成 Complex SMILES (全显式) ---
        # 保持原样，不合并 H
        c_smiles = Chem.MolToSmiles(mol)
        complex_list.append(c_smiles)

        # --- 生成 Simple SMILES (折叠式) ---
        # 使用 RemoveHs 将独立 H 节点折叠为原子的属性
        try:
            # 必须先做一个副本，避免修改原 mol 影响 complex_list
            mol_copy = Chem.Mol(mol)
            # RemoveHs 会根据成键关系自动处理隐式氢
            simple_mol = Chem.RemoveHs(mol_copy)

            # allHsExplicit=True 确保显示为 [CH4] 而不是 C
            # 这对电催化中间体非常重要，能一眼看出质子化程度
            s_smiles = Chem.MolToSmiles(simple_mol, allHsExplicit=True)
            simple_list.append(s_smiles)
        except Exception:
            # 万一折叠失败（例如非标准价态），回退到原始输出
            simple_list.append(c_smiles)

    # 排序确保顺序无关性（例如 A+B 和 B+A 视为同一状态）
    # 使用 " . " 连接符合 RDKit 的多组分 SMILES 标准
    # final_simple = " . ".join(sorted(simple_list))
    # final_complex = " . ".join(sorted(complex_list))

    return simple_list, complex_list


def collect_paths_from_nodes(end_nodes: List[SearchNode]) -> List[List[str]]:
    """
    将搜索到的终点节点列表转化为绘图所需的签名路径列表。
    """
    all_paths = []
    for node in end_nodes:
        path_signatures = []
        # 使用你定义的 iter_path() 回溯
        for step_node in node.iter_path():
            signature = get_state_signature(step_node.state)
            path_signatures.append(signature[0][0])
        all_paths.append(path_signatures)
    return all_paths




def save_search_results(found_paths: list, filename: str = "debug_paths.pkl"):
    """保存搜索结果的所有节点对象"""
    with open(filename, "wb") as f:
        pickle.dump(found_paths, f)
    print(f"✅ 原始路径数据已保存至 {filename}")

def load_search_results(filename: str = "debug_paths.pkl"):
    """加载保存的路径数据"""
    with open(filename, "rb") as f:
        return pickle.load(f)


