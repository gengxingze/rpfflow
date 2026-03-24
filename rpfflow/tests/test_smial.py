import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

# -----------------------------
# 1. 设置文件路径
# -----------------------------
# 假设你的文件名如下，请根据实际情况修改
file_updated = "../../path_Ag_updated.extxyz"
file_result = "../../path_result_Ag.extxyz"

def get_energies(filename):
    """读取文件并提取所有帧的能量"""
    configs = read(filename, index=":")
    energies = []
    for atoms in configs:
        # 优先从 info 中取，如果不存在则尝试从 calculator 取
        if 'energy' in atoms.info:
            energies.append(atoms.info['energy'])
        else:
            try:
                energies.append(atoms.get_potential_energy())
            except:
                print(f"警告：文件 {filename} 中某帧缺失能量信息")
    return energies

# -----------------------------
# 2. 读取数据
# -----------------------------
e_updated = np.array(get_energies("../../path_Ag_updated.extxyz") + get_energies("../../path_Cu_updated.extxyz") + get_energies("../../path_Pt_updated.extxyz"))
e_result = np.array(get_energies("../../path_result_Ag.extxyz") + get_energies("../../path_result_Cu.extxyz") + get_energies("../../path_result_Pt.extxyz"))

# 检查数据长度是否一致
if len(e_updated) != len(e_result):
    print(f"警告：两个文件的帧数不一致！({len(e_updated)} vs {len(e_result)})")
    # 取较短的长度进行对齐绘图
    min_len = min(len(e_updated), len(e_result))
    e_updated = e_updated[:min_len]
    e_result = e_result[:min_len]

# -----------------------------
# 3. 绘制对角图
# -----------------------------
plt.figure(figsize=(12, 10))

# 绘制散点图
plt.scatter(e_updated, e_result, alpha=0.6, edgecolors='k', label='Data Points')

# 绘制 y=x 参考线
all_values = np.concatenate([e_updated, e_result])
min_val, max_val = all_values.min(), all_values.max()
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')

# 计算 RMSE 或 MAE (可选)
rmse = np.sqrt(np.mean((e_updated - e_result)**2))
plt.text(min_val, max_val, f'RMSE: {rmse:.4f} eV', verticalalignment='top')

# 图形修饰
plt.xlabel('Energy from DFT (eV)')
plt.ylabel('Energy from MACE (eV)')
plt.title('Energy Parity Plot')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal') # 确保横纵坐标比例一致

plt.tight_layout()
plt.savefig("parity_plot.png", dpi=300) # 如果需要保存图片
plt.show()



