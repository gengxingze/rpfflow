import matplotlib.pyplot as plt
import numpy as np


def plot_energy_profile(energy_list, labels, title="Reaction Energy Profile", colors=None):
    """
    绘制不带过渡态的反应能量图

    :param energy_list: 能量列表。如果是多组对比，请传入列表的列表，如 [[0, -0.5, 0.2], [0, -0.8, 0.1]]
    :param labels: 对应每个台阶的化学式标签列表
    :param title: 图表标题
    :param colors: 颜色列表，用于区分不同组数据
    """

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

    plt.tight_layout()
    plt.show()


# --- 使用示例 ---
my_labels = ["$CH_2^* + CH_2CCH_3^*$", "$CH_2CH_2CCH_3^*$", "$CH_2CH_2CHCH_3^*$", "$CH_2CH_2CH_2CH_3^*$", "$butane$"]
# 模拟 0V 和 -0.8V 的能量数据
data_0V = [0.0, -0.32, -0.81, -0.13, -0.25]
data_neg08V = [0.0, -0.45, -0.95, -0.20, -0.35]

# 调用函数
plot_energy_profile([data_0V, data_neg08V], my_labels, title="Comparison of Different Voltages")