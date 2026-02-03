"""
mol_visualization.py

Visualization utilities for:
- NetworkX molecular graphs
- RDKit 2D molecules
- RDKit 3D molecules (py3Dmol)

Author: Xingze Geng
"""

from typing import List, Optional, Callable
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


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


def plot_reaction_tree(paths, direction='LR'):
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
            font_size=11,  # 稍微缩小字体以适应长 SMILES
            edge_color='#90A4AE',
            width=2.0,
            node_shape='o',  # 方框更适合放长 SMILES 标签
            bbox=dict(facecolor='white', edgecolor='#1976D2', boxstyle='round,pad=0.3'))

    plt.title(f"Reaction Tree", fontsize=16, pad=20)

    # 自动保存为高分辨率图片以便缩放查看
    plt.savefig("reaction_path.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_energy_profile(energy_list, labels, title="Reaction Energy Profile", colors=None):
    """
    绘制不带过渡态的反应能量图

    :param energy_list: 能量列表。如果是多组对比，请传入列表的列表，如 [[0, -0.5, 0.2], [0, -0.8, 0.1]]
    :param labels: 对应每个台阶的化学式标签列表
    :param title: 图表标题
    :param colors: 颜色列表，用于区分不同组数据
    """
    import matplotlib
    matplotlib.use('TkAgg')
    # 统一处理数据格式：确保 energy_list 是嵌套列表
    if not isinstance(energy_list[0], (list, np.ndarray)):
        energy_list = [energy_list]

    if colors is None:
        colors = ['black', '#1f77b4', '#ff7f0e', '#2ca02c']

    # 绘图参数
    step_width = 0.6  # 平台宽度
    gap = 0.4  # 平台间距

    plt.figure(figsize=(12, 6))

    # 遍历每一组能量数据
    for group_idx, energies in enumerate(energy_list):
        color = colors[group_idx % len(colors)]
        x_start = 0

        for i in range(len(energies)):
            x_range = [x_start, x_start + step_width]

            # 1. 绘制水平能级
            plt.hlines(energies[i], x_range[0], x_range[1], color=color, lw=2.5,
                       label=f"Group {group_idx + 1}" if i == 0 else "")

            # 2. 仅在第一组数据上标注标签（避免重复）
            if group_idx == 0:
                plt.text(x_start + step_width / 2, min(energies) - 0.3, labels[i],
                         ha='center', va='top', fontsize=10, fontweight='bold', rotation=15)

            # 3. 绘制连接虚线
            if i > 0:
                prev_x_end = x_start - gap
                plt.plot([prev_x_end, x_start], [energies[i - 1], energies[i]],
                         color=color, linestyle=':', lw=1.5, alpha=0.7)

            x_start += (step_width + gap)

    # 图表修饰
    plt.axhline(0, color='gray', lw=0.8, ls='--')  # 零能级参考线
    plt.ylabel('$\Delta E$ (eV)', fontsize=12)
    plt.xlabel('Reaction Coordinate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks([])

    # 如果有多组数据，显示图例
    if len(energy_list) > 1:
        plt.legend()

    # plt.tight_layout()
    plt.savefig("ReactionCoordinate.png", dpi=300)
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
    from rpfflow.core.structure import process_extxyz_energies
    aa = process_extxyz_energies("../core/path_0.extxyz")
    plot_energy_profile(aa[0], aa[1])
    print("✓ Visualization done.")


