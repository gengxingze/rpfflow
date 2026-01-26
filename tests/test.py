from ase.io import read
import numpy as np

# 读取所有构型
images = read("../test.extxyz", ":")

energies = [atoms.get_potential_energy()for atoms in images]

print("构型数目:", len(energies))
print("前几个能量:", energies[:5])

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

n = len(energies)
plt.figure(figsize=(6,4))

# ---------- 1. 画水平实线（平台） ----------
for i in range(n):
    plt.plot([i, i+1], [energies[i], energies[i]],
             linestyle='-', linewidth=2)

# ---------- 2. 画竖直虚线（跃迁） ----------
for i in range(1, n):
    plt.plot([i, i], [energies[i-1], energies[i]],
             linestyle='--', linewidth=1)

# ---------- 3. 结构中心点 ----------
x_center = np.arange(n) + 0.5
plt.plot(x_center, energies, "o", ms=5)

plt.xlim(0, n)
plt.xticks(x_center, range(n))

plt.xlabel("Structure index")
plt.ylabel("Energy (eV)")
plt.title("Energy step profile")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../test.png")
plt.show()

print("sucessful")
