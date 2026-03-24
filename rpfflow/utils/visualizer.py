"""
mol_visualization.py

Visualization utilities for:
- NetworkX molecular graphs
- RDKit 2D molecules
- RDKit 3D molecules (py3Dmol)

Author: Xingze Geng
"""
import os
from typing import List, Optional, Callable
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

plt.rcParams.update({'font.size': 18, 'font.family': 'serif', 'font.serif': ['Times New Roman']})
plt.rcParams['mathtext.default'] = 'regular'

# ============================================================
# Visualization configuration
# ============================================================

ELEMENT_COLORS = {
    "H": "#CCCCCC",
    "C": "#4C4C4C",
    "O": "#E74C3C",
    "N": "#3498DB",
    "S": "#F1C40F",
    "P": "#9B59B6",
    "F": "#1ABC9C",  # virtual / active site
}


# ============================================================
# NetworkX molecular graph visualization
# ============================================================

def plot_molecular_graph(
    G: nx.Graph,
    title: str = "Molecular Graph",
    save_path: Optional[str] = None,
    layout: Callable = nx.spring_layout,
    node_size: int = 800,
    font_size: int = 12,
) -> None:
    """
    Plot a molecular graph using NetworkX + Matplotlib.

    Required node attribute:
        - symbol

    Optional edge attribute:
        - bond_order
    """
    plt.rcParams["font.family"] = "Times New Roman"

    try:
        node_colors = [
            ELEMENT_COLORS.get(G.nodes[n].get("symbol", "R"), "#BDC3C7")
            for n in G.nodes
        ]

        labels = {n: G.nodes[n].get("symbol", str(n)) for n in G.nodes}
        pos = layout(G, seed=159632)

        plt.figure(figsize=(6, 6))
        nx.draw(
            G,
            pos,
            labels=labels,
            node_color=node_colors,
            node_size=node_size,
            font_size=font_size,
            with_labels=True,
            edgecolors="black",
            linewidths=1.2,
            width=1.6,
        )

        edge_labels = {
            (u, v): d.get("bond_order", "")
            for u, v, d in G.edges(data=True)
        }

        if edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, font_size=font_size - 2
            )

        plt.title(title)
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        plt.show()

    except Exception as e:
        plt.close()
        raise RuntimeError(f"Graph visualization failed: {e}")


def plot_molecular_graphs(
    graphs: List[nx.Graph],
    titles: Optional[List[str]] = None,
    cols: int = 3,
    save_path: Optional[str] = None,
    layout: Callable = nx.spring_layout,
    node_size: int = 800,
    font_size: int = 12,
) -> None:
    """
    Plot multiple molecular graphs in a grid layout.
    """
    plt.rcParams["font.family"] = "Times New Roman"

    n = len(graphs)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
    axes = np.atleast_2d(axes)

    for i, G in enumerate(graphs):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.set_axis_off()

        node_colors = [
            ELEMENT_COLORS.get(G.nodes[n].get("symbol", "R"), "#BDC3C7")
            for n in G.nodes
        ]

        labels = {n: G.nodes[n].get("symbol", str(n)) for n in G.nodes}
        pos = layout(G, seed=159632)

        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            labels=labels,
            node_color=node_colors,
            node_size=node_size,
            font_size=font_size,
            edgecolors="black",
            linewidths=1.2,
            width=1.6,
        )

        edge_labels = {
            (u, v): d.get("bond_order", "")
            for u, v, d in G.edges(data=True)
        }

        if edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, font_size=font_size - 2, ax=ax
            )

        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=font_size + 2)
        else:
            ax.set_title(f"Molecule {i+1}", fontsize=font_size + 2)

    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


# ============================================================
# RDKit molecule visualization
# ============================================================

def show_molecule_3d(
    mol: Chem.Mol,
    style: str = "stick",
    width: int = 400,
    height: int = 400,
    export_html: Optional[str] = None,
):
    """
    Visualize an RDKit Mol in 3D using py3Dmol.
    """
    import py3Dmol
    mol_block = Chem.MolToMolBlock(Chem.Mol(mol))

    view = py3Dmol.view(width=width, height=height)
    view.addModel(mol_block, "mol")
    view.setStyle({style: {}})
    view.zoomTo()
    view.show()

    if export_html:
        html = view._make_html()
        with open(export_html, "w") as f:
            f.write(html)

    return view


def save_molecule_2d(
        mol: Chem.Mol,
        filename: str = "molecule.png",
        size=(500, 500),
) -> None:
    """
        通过手动设置 atomLabel 属性，强制显示所有碳原子符号。
        """
    # 1. 必须先做一个副本，以免修改原始分子对象
    temp_mol = Chem.Mol(mol)

    # 2. 遍历所有原子，强制给碳原子打上 "C" 标签
    for atom in temp_mol.GetAtoms():
        if atom.GetSymbol() == "C":
            # SetProp 会直接覆盖绘图时的默认隐藏逻辑
            atom.SetProp("atomLabel", "C")

    # 3. 准备绘图坐标
    if not temp_mol.GetNumConformers():
        Draw.PrepareMolForDrawing(temp_mol)

    # 4. 创建绘图对象
    drawer = Draw.MolDraw2DCairo(*size)

    # 既然 updateFromDict 报错，我们尝试逐个尝试最可能的属性名
    # 有的版本叫 explicitMethylGroup（显式甲基），可以一试
    opts = drawer.drawOptions()
    try:
        opts.prepareMolsBeforeDrawing = True
    except:
        pass

    # 5. 执行绘图
    drawer.DrawMolecule(temp_mol)
    drawer.FinishDrawing()

    # 6. 保存
    with open(filename, "wb") as f:
        f.write(drawer.GetDrawingText())


def plot_reaction_tree(paths, direction='LR', file_name='reaction_tree.png'):
    G = nx.DiGraph()
    node_depths = {}
    layer_counts = {}

    # 1. 构建图并统计每层的节点数
    for path in paths:
        for i, node_name in enumerate(path):
            G.add_node(node_name)
            if node_name not in node_depths:
                node_depths[node_name] = i
                layer_counts[i] = layer_counts.get(i, 0) + 1
            else:
                node_depths[node_name] = min(node_depths[node_name], i)
        G.add_edges_from(zip(path, path[1:]))

    for node, depth in node_depths.items():
        G.nodes[node]['layer'] = depth

    # 2. 动态计算画布大小 (核心改进)
    max_depth = max(node_depths.values()) if node_depths else 1
    max_nodes_in_layer = max(layer_counts.values()) if layer_counts else 1

    # 根据节点数量设定比例因子 (每个节点大约占用 2.5 - 3 英寸)
    scale = 1.0
    if direction == 'TB':
        width = (max_depth + 1) * scale * 1.5  # 深度决定宽度
        height = max_nodes_in_layer * scale  # 最宽的一层决定高度
    else:
        width = max_nodes_in_layer * scale
        height = (max_depth + 1) * scale * 1.5

    # 限制最小和最大尺寸，防止太小或超出内存
    fig_size = (max(min(width, 50), 10), max(min(height, 50), 6))

    # 3. 布局与绘图
    plt.figure(figsize=fig_size)
    pos = nx.multipartite_layout(G, subset_key='layer', align='vertical' if direction == 'LR' else 'horizontal')

    if direction == 'LR':
        pos = {node: [coords[1], -coords[0]] for node, coords in pos.items()}

    # 4. 优化显示细节
    nx.draw(G, pos,
            with_labels=True,
            node_color='#E3F2FD',  # 浅蓝色调，更专业
            node_size=4000,  # 增大节点
            arrowstyle='-|>',
            arrowsize=25,
            font_weight='bold',
            font_color='black',
            font_family="Times New Roman",
            font_size=11,  # 稍微缩小字体以适应长 SMILES
            edge_color='#90A4AE',
            width=2.0,
            node_shape='o',  # 方框更适合放长 SMILES 标签
            bbox=dict(facecolor='white', edgecolor='#1976D2', boxstyle='round,pad=0.3'))

    plt.title(f"Reaction Tree", fontsize=16, pad=20)

    # 自动保存为高分辨率图片以便缩放查看
    plt.savefig(file_name, dpi=300)
    plt.show()


def draw_reaction_networks(all_paths, highlight_index=0, output_name="reaction_network", format="png"):
    """
    根据路径列表绘制反应网络图

    Parameters
    ----------
    all_paths : List[Tuple[str]]
        每条路径是一个 tuple，如:
        ('A', 'B', 'C', ...)
    highlight_index : int
        高亮哪一条路径（主路径）
    output_name : str
        输出文件名
    format : str
        输出格式 (png/pdf/svg)
    """
    os.environ["PATH"] += os.pathsep + r'D:\Program\Graphviz\bin'
    from graphviz import Digraph
    dot = Digraph(format=format)

    # ================= 全局样式 =================
    dot.attr(rankdir="TB", splines="spline", nodesep="0.5", ranksep="0.6", dpi='300')
    dot.attr("node",
             shape="box",
             style="rounded,filled",
             fontname="Helvetica",
             fontsize="11",
             fillcolor="#FAFAFA",
             color="#4A90E2")

    dot.attr("edge",
             color="#666666",
             arrowsize="0.7",
             penwidth="1.1",
             fontname="Helvetica")

    # ================= 收集所有节点 =================
    nodes = set()
    edges = set()

    for path in all_paths:
        for i in range(len(path)):
            nodes.add(path[i])
            if i < len(path) - 1:
                edges.add((path[i], path[i + 1]))

    # ================= 画节点 =================
    for node in nodes:

        # 起点
        if any(node == p[0] for p in all_paths):
            dot.node(node, fillcolor="#2C3E50", fontcolor="white", color="#1A252F", penwidth="2")

        # 终点
        elif any(node == p[-1] for p in all_paths):
            dot.node(node, shape="ellipse", fillcolor="#E8F5E9", color="#2E7D32", penwidth="2")

        # 关键中间体（例：CH3）
        elif node == "[CH3]":
            dot.node(node, fillcolor="#FFF3E0", color="#E65100", penwidth="1.8")

        else:
            dot.node(node)

    # ================= 主路径 =================
    # main_path = all_paths[highlight_index]
    # main_edges = set((main_path[i], main_path[i + 1])
    #                  for i in range(len(main_path) - 1))

    # ================= 画边 =================
    for u, v in edges:
        # if (u, v) in main_edges:
        #     dot.edge(u, v, color="#000000", enwidth="2.0")
        # else:
            dot.edge(u, v, style="dashed", color="#999999")

    # ================= 输出 =================
    dot.render(output_name, view=False)
    return dot

def plot_energy_profile(
        energy_list,
        states=None,
        mode="sequence",   # "sequence" or "aligned"
        title="Energy Diagram",
        colors=None,
        labels=None,
        show_values=True,
        step_width=0.6,
        gap=0.4,
        save_path=None):
    """
    通用能量图绘制函数

    Parameters
    ----------
    energy_list : list of list
    states : list (sequence模式) 或 list of list (aligned模式)
    mode : "sequence" | "aligned"
    """

    # -------- 输入统一 --------
    if not isinstance(energy_list[0], (list, tuple, np.ndarray)):
        energy_list = [energy_list]

    if colors is None:
        colors = ['#08519c', '#a50f15', '#006d2c', '#54278f']

    fig, ax = plt.subplots(figsize=(12, 6))

    # =========================================================
    # 🔷 模式2：对齐模式（StepChartPlotter）
    # =========================================================
    if mode == "aligned":
        data_list = []
        for i in range(len(energy_list)):
            data_list.append({"states": states[i], "energies": energy_list[i], "label": labels[i]})

        # --- 1. 核心改进：计算全局最优排序 (拓扑对齐) ---
        # 我们需要一个列表来存储所有状态，顺序尽量符合所有路径的先后
        global_order = []
        for entry in data_list:
            states = entry['states']
            for i, s in enumerate(states):
                if s not in global_order:
                    # 寻找插入位置：放在上一个已知状态的后面
                    if i == 0:
                        global_order.insert(0, s)
                    else:
                        prev_s = states[i - 1]
                        idx = global_order.index(prev_s)
                        global_order.insert(idx + 1, s)
                else:
                    # 如果已经在里面了，检查是否违背了当前的先后顺序
                    # (处理 A-B-D-C 这种逻辑，此处可根据需要做更复杂的 DAG 排序)
                    pass
        # 建立映射
        state_to_x = {s: i for i, s in enumerate(global_order)}

        # --- 2. 绘图 ---
        for p_idx, entry in enumerate(data_list):
            color = colors[p_idx % len(colors)]
            states = entry['states']
            energies = entry['energies']

            # 记录每一步的端点，用于连线
            path_coords = []
            already_states = []

            for i in range(len(states)):
                s = states[i]
                e = energies[i]
                x_center = state_to_x[s]
                # 绘制台阶
                x_start, x_end = x_center - step_width / 2, x_center + step_width / 2
                # 台阶
                ax.hlines(e, x_start, x_end, color=color, lw=3.5, zorder=3, label=labels[p_idx] if (labels and i == 0) else None)

                # 数值
                if show_values:
                    ax.text((x_start + x_end)/2, e + 0.03,f"{e:.2f}", ha='center', va="bottom", fontsize=9, color=color)

                # 状态标签（只画一次）
                if states is not None and states not in already_states:
                    y = e - 0.2

                    ax.text((x_start + x_end) / 2,y,states[i],ha='center',va='top',fontsize=8,fontweight='regular',
                        style='italic'
                    )
                    already_states.append(states[i])
                # 存储坐标信息
                path_coords.append({'x_start': x_start, 'x_end': x_end, 'y': e})

            # 连接线
            # 绘制连线
            for i in range(len(path_coords) - 1):
                curr = path_coords[i]
                nxt = path_coords[i + 1]

                # 检查是否是“回头路”
                ls = '--' if nxt['x_start'] >= curr['x_end'] else '-.'
                alpha = 0.6 if ls == '-' else 0.3  # 回头路用淡虚线

                ax.plot([curr['x_end'], nxt['x_start']], [curr['y'], nxt['y']],
                        color=color, lw=1.5, ls=ls, alpha=alpha, zorder=2)
        ax.set_xticklabels([])
    # =========================================================
    # 🔷 模式1：顺序模式（你现在这个函数）
    # =========================================================
    elif mode == "sequence":

        for g_idx, energies in enumerate(energy_list):
            color = colors[g_idx % len(colors)]
            for i, e in enumerate(energies):
                x_start = i * (step_width + gap)
                x_end = x_start + step_width
                # 台阶
                ax.hlines(e, x_start, x_end, color=color, lw=3.5, zorder=3, label=labels[g_idx] if (labels and i == 0) else None)

                # 数值
                if show_values:
                    ax.text((x_start + x_end)/2, e + 0.03,f"{e:.2f}", ha='center', fontsize=9, color=color)

                # 状态标签（只画一次）
                if states is not None and g_idx == 0:
                    all_e = [g[i] for g in energy_list if i < len(g)]
                    ax.text((x_start + x_end)/2, min(all_e) - 0.15,states[i], ha='center', fontsize=8)

                # 连接线
                if i > 0:
                    prev_x = (i - 1) * (step_width + gap) + step_width
                    ax.plot([prev_x, x_start],[energies[i-1], e], color=color, ls='--', lw=1.2, alpha=0.7, zorder=2)


    else:
        raise ValueError("mode 必须是 sequence 或 aligned")

    # -------- 通用美化 --------
    ax.axhline(0, color='#636363', lw=1, ls='-', alpha=0.3, zorder=1)  # 零能级参考线
    # 坐标轴标签
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_ylabel('Energy (eV)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Reaction Coordinate', fontsize=16, fontweight='bold')

    if mode == "sequence":
        ax.set_xticks([])

    ax.grid(axis='y', linestyle=':', alpha=0.4)

    if labels:
        ax.legend(frameon=False, loc="best", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    # from convert import (
    #     create_mol,
    #     rdkit_to_nx,
    #     generate_coordinate,
    #     create_common_molecules,
    #     nx_to_rdkit,
    # )
    #
    # mol = create_mol("O=C(O)C(=O)O", add_h=False)
    # G = rdkit_to_nx(mol)
    # from rpfflow.core.state import GraphState
    # G = GraphState(graph=[G])
    # G.update()
    # plot_molecular_graph(G.graph[0], title="Formic acid", save_path="formic_graph.png")
    #
    # templates = create_common_molecules()
    # mol_tmp = generate_coordinate(nx_to_rdkit(templates["OCRO"]))
    # save_molecule_2d(mol_tmp)
    # save_molecule_2d(mol, filename="../../graph_mm/test.png")
    #



    # Ag_DFT = process_extxyz_energies("../../path_Ag_updated.extxyz")
    # Cu_DFT = process_extxyz_energies("../../path_Cu_updated.extxyz")
    # Pt_DFT = process_extxyz_energies("../../path_Pt_updated.extxyz")
    # plot_energy_profile([Ag_DFT[2], Cu_DFT[2], Pt_DFT[2]], states=Ag_DFT[1], labels=["DFT-Ag", "DFT-Cu", "DFT-Pt"], mode="sequence", title="Energy Diagram")

    Cu = process_extxyz_energies("../../path_result_Cu.extxyz")
    Cu_1 = process_extxyz_energies("../../path_result_Cu_1.extxyz")
    Cu_2 = process_extxyz_energies("../../path_result_Cu_2.extxyz")
    plot_energy_profile([Cu[2], Cu_1[2], Cu_2[2]], states=[Cu[1], Cu_1[1], Cu_2[1]], labels=["path=1", "path=2", "path=3"],
                        mode="aligned",save_path="EnergyProfile_Cu.png")

    print("✓ Visualization done.")


