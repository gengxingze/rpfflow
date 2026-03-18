import logging
import os
import torch
import numpy as np
import networkx as nx
from copy import deepcopy
from typing import Dict, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.db import connect
from collections import Counter
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from rpfflow.utils.convert import rdkit_to_nx


logger = logging.getLogger(__name__)

# ============================================================
# RDKit molecule construction
# ============================================================
def create_mol(
    smiles: Optional[str] = None,
    structure: Optional[str] = None,
    add_h: bool = False,
    optimize: bool = False
) -> Chem.Mol:
    """
    Create an RDKit molecule from SMILES or structure file.

    Args:
        smiles: SMILES string
        structure: mol file path
        add_h: whether to add hydrogens
        optimize: whether to run UFF optimization

    Returns:
        RDKit Mol object with 3D conformer.
    """
    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
    elif structure is not None:
        mol = Chem.MolFromMolFile(structure)
    else:
        raise ValueError("Either smiles or structure must be provided.")

    if mol is None:
        raise ValueError("Failed to create RDKit Mol.")

    if add_h:
        mol = Chem.AddHs(mol)

    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL, catchErrors=True)

    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if optimize:
        AllChem.UFFOptimizeMolecule(mol)

    return mol


def generate_coordinate(mol: Chem.Mol) -> Chem.Mol:
    """
    Safely generate 3D coordinates without enforcing full sanitization.
    Suitable for TS or broken structures.
    """
    if not isinstance(mol, Chem.Mol):
        raise TypeError("Input must be an RDKit Mol object.")

    mol = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SANITIZE_SETAROMATICITY |
                        Chem.SANITIZE_FINDRADICALS |
                        Chem.SANITIZE_SETCONJUGATION
        )
    except Exception as e:
        print("[WARN] Partial sanitize failed (allowed):", e)
        mol.UpdatePropertyCache(strict=False)

    if mol.GetNumConformers() == 0:
        params = AllChem.ETKDG()
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError("3D embedding failed.")

    return mol


# ============================================================
# Common molecular graph templates
# ============================================================
def create_common_molecules() -> Dict[str, nx.Graph]:
    """
    Build common molecular graph templates.
    """
    molecules = {}

    smiles_dict = {
        "OH_": "[OH-]",
        "H": "[H]",
        "H2O": "O",
        "CO2": "O=C=O",
    }

    for name, smi in smiles_dict.items():
        mol = create_mol(smi, add_h=True, optimize=True)
        molecules[name] = rdkit_to_nx(mol)
        print(f"[✓] {name} generated.")

    from rpfflow.rules.basica import update_valence

    # O=C-O-R
    G1 = nx.Graph()
    G1.add_nodes_from([
        (0, {"symbol": "O"}),
        (1, {"symbol": "C"}),
        (2, {"symbol": "O"}),
        (3, {"symbol": "F"}),  # active site
    ])
    G1.add_edge(0, 1, bond_order=2.0)
    G1.add_edge(1, 2, bond_order=1.0)
    G1.add_edge(2, 3, bond_order=1.0)
    update_valence(G1)
    molecules["OCOR"] = G1

    # O=C(R)-O
    G2 = nx.Graph()
    G2.add_nodes_from([
        (0, {"symbol": "O"}),
        (1, {"symbol": "C"}),
        (2, {"symbol": "O"}),
        (3, {"symbol": "F"}),
    ])
    G2.add_edge(0, 1, bond_order=2.0)
    G2.add_edge(1, 2, bond_order=1.0)
    G2.add_edge(1, 3, bond_order=1.0)
    update_valence(G2)
    molecules["OCRO"] = G2

    R = deepcopy(molecules["H"])
    R.nodes[0]["symbol"] = "F"
    update_valence(R)
    molecules["F"] = R

    return molecules



def rotate_F(atoms: Atoms):
    pos = atoms.get_positions()

    # === 1. 找 F 原子 ===
    symbols = atoms.get_chemical_symbols()
    if "F" not in symbols:
        raise ValueError("结构中没有 F 原子")
    iF = symbols.index("F")
    rF = pos[iF]

    # === 2. 找最近相连原子 ===
    dists = np.linalg.norm(pos - rF, axis=1)
    dists[iF] = 1e10
    iN = np.argmin(dists)
    rN = pos[iN]

    v = rN - rF
    v = v / np.linalg.norm(v)

    z = np.array([0.0, 0.0, 1.0])

    # === 3. 计算旋转矩阵 (Rodrigues) ===
    axis = np.cross(v, z)
    sin_theta = np.linalg.norm(axis)
    cos_theta = np.dot(v, z)

    if sin_theta < 1e-8:
        R = np.eye(3)
    else:
        axis = axis / sin_theta
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + K * sin_theta + K @ K * (1 - cos_theta)

    # === 4. 以 F 为中心旋转 ===
    pos_rot = (pos - rF) @ R.T + rF
    atoms.set_positions(pos_rot)

    # === 5. 保证 F 的 Z 最小 ===
    z_coords = atoms.get_positions()[:, 2]
    if atoms.positions[iF, 2] > np.min(z_coords):
        atoms.positions[:, 2] *= -1

    return atoms



def optimize_structure(atoms, model_size="small", fmax=0.05, steps=200,
                       fix_mask=None, db_path="calculations.db"):
    """
    带有双重信息校验（Formula & Fragment Signature）的结构优化持久化函数
    """

    # 1. 提取双重校验信息
    formula = atoms.get_chemical_formula()
    # 使用 .get() 避免 Key 缺失导致报错
    signature = atoms.info.get('fragment_signature', 'unknown')

    # 2. 数据库查询：双重命中检测
    if os.path.exists(db_path):
        with connect(db_path) as db:
            # 这里的 signature 以自定义 key 的形式存储在数据库的 data 或 key_value_pairs 中
            # 我们根据 formula 缩小范围，再比对 signature
            for row in db.select(formula=formula):
                # row.key_value_pairs 会存储我们写入的自定义属性
                db_signature = row.get('signature')

                if db_signature == signature:
                    # 如果受力要求也满足，则直接返回
                    if row.get('fmax', 0.05) <= fmax:
                        # logger.info(f"双重校验命中: {formula} [{signature}]")
                        cached_atoms = row.toatoms()
                        # 注意：从 DB 恢复后需重新绑定计算器，否则无法获取能量
                        # 或者直接使用 row.energy
                        return cached_atoms

    # 3. 准备计算环境 (仅在未命中的情况下执行)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 避免重复加载模型：如果 atoms 已经有合适的 calc，就不再初始化
    if not hasattr(atoms, 'calc') or atoms.calc is None:
        from mace.calculators import mace_mp
        calc = mace_mp(model=model_size, device=device, default_dtype="float32")
        atoms.calc = calc

    if fix_mask is not None:
        atoms.set_constraint(FixAtoms(mask=fix_mask))

    # 4. 执行优化
    # 使用 logfile=None 避免产生大量无用的 opt.log 文件
    dyn = BFGS(atoms, logfile=None)
    dyn.run(fmax=fmax, steps=steps)

    # -------- 5. 写入前校验（关键）--------
    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        atoms.calc.results['forces'] = forces.astype(np.float64)
        if forces.shape != (len(atoms), 3):
            raise RuntimeError("Invalid forces shape")

    except Exception as e:
        # 不写入坏数据
        print("Skip DB write due to invalid calculation:", e)
        return atoms

    # -------- 6. 防炸写入（核心）--------
    with connect(db_path) as db:
        db.write(
            atoms,
            signature=signature,
            model=model_size
        )

        db.connection.commit()  # 强制写盘

    return atoms


def generate_adsorption_structures(adsorbate_ase, slab_ase, repeat=(1, 1, 1),min_lw=4.0):
    """
    Generate adsorption structures using pymatgen, input/output in ASE Atoms.

    Parameters
    ----------
    slab_ase : ase.Atoms
        Slab structure (with vacuum).
    adsorbate_ase : ase.Atoms
        Adsorbate molecule.
    repeat : tuple
        Supercell for adsorption search.
    min_lw : float
        Minimum vacuum thickness.

    Returns
    -------
    ads_structures_ase : list of ase.Atoms
        Generated adsorption structures.
    ads_sites : dict
        Adsorption sites from pymatgen (ontop/bridge/hollow/all).
    """

    adaptor = AseAtomsAdaptor()

    # ASE -> pymatgen
    slab_pm = adaptor.get_structure(slab_ase)
    from pymatgen.core import Molecule
    ads_species = adsorbate_ase.get_chemical_symbols()
    ads_coords = adsorbate_ase.get_positions()
    ads_pm = Molecule(species=ads_species, coords=ads_coords)

    # 构建吸附位点搜索器
    asf = AdsorbateSiteFinder(slab_pm)

    # 找位点
    ads_sites = asf.find_adsorption_sites()


    ads_mols = [ads_pm]

    # 生成吸附结构
    ads_structures_pm = []
    for mol in ads_mols:
        ads_structures_pm += asf.generate_adsorption_structures(
            mol,
            repeat=list(repeat),
            min_lw=min_lw,
        )

    def pmg_to_ase_drop_tags(struct):
        struct = struct.copy()

        # 删 structure 级别
        struct.site_properties.pop("tags", None)

        # 删 site 级别（关键）
        for site in struct.sites:
            if "tags" in site.properties:
                site.properties.pop("tags", None)
            if "tag" in site.properties:
                site.properties.pop("tag", None)

        return adaptor.get_atoms(struct)
    # pymatgen -> ASE
    ads_structures_ase = [pmg_to_ase_drop_tags(s) for s in ads_structures_pm]


    return ads_structures_ase, ads_sites


def get_reference_structure(slab_ase):
    from ase.build import molecule
    """
    创建参考分子（H, H2O）并进行几何优化。
    返回一个包含优化后 ASE 对象和原始 Slab 的字典。
    """
    # 1. 创建初始结构
    # 'H' 原子的 potential_energy 建议直接设为参考值，或者创建孤立原子
    h_atom = molecule('H')
    # h_atom.set_cell([10, 10, 10])  # 给孤立原子一个足够大的真空层
    h_atom.center()

    h2o_mol = molecule('H2O')
    # h2o_mol.set_cell([12, 12, 12])
    h2o_mol.center()

    # 2. 几何优化
    # 注意：孤立原子的 optimize_structure 可能会因为没有受力而跳过
    print("--- 优化参考结构 H ---")
    opt_h = optimize_structure(h_atom, fmax=0.05, steps=200)

    print("--- 优化参考结构 H2O ---")
    opt_h2o = optimize_structure(h2o_mol, fmax=0.05, steps=200)

    opt_f = optimize_structure(slab_ase, fmax=0.05, steps=200)
    # 3. 构建结果字典
    reference_dict = {
        "H": opt_h,
        "H2O": opt_h2o,
        "F": opt_f  # 将传入的 slab 作为 F 的参考（对应表面）
    }

    return reference_dict


def process_extxyz_energies(filename):
    """
    读取extxyz，按step合并能量并拼接化学式
    """
    from collections import defaultdict
    from ase.io import read
    # 用于存储数据：{step: {'energy': total_energy, 'formulas': [formula1, formula2, ...]}}
    step_data = defaultdict(lambda: {'energy': 0.0, 'formulas': []})

    # 1. 加载所有结构
    # index=':' 表示读取文件中所有的构型
    configs = read(filename, index=':')

    for atoms in configs:
        # 获取当前构型的 step (从 atoms.info 字典中读取)
        # 如果 atoms.info 中没有 'step'，则默认为 0
        step = atoms.info.get('step', 0)

        # 获取能量 (假设能量存储在 atoms.info['energy'])
        # 如果 extxyz 里没有能量字段，可能需要 atoms.get_potential_energy()
        energy = atoms.get_potential_energy()

        # 获取化学式
        try:
            formula = atoms.info['fragment_signature']
        except:
            formula = atoms.get_chemical_formula()

        # 累加能量并记录化学式
        step_data[step]['energy'] += energy
        step_data[step]['formulas'].append(formula)

    # 2. 整理输出
    sorted_steps = sorted(step_data.keys())
    final_energies = []
    final_labels = []

    for s in sorted_steps:
        total_e = step_data[s]['energy']
        # 将化学式用 '+' 连接
        combined_formula = " + ".join(step_data[s]['formulas'])

        final_energies.append(total_e)
        final_labels.append(combined_formula)

        print(f"Step {s}: Energy = {total_e:.4f} eV, Formula = {combined_formula}")

    return final_energies, final_labels


def compute_adsorption_energy(reactants, products):
    """
    计算吸附能：

    E_ads = E_products - E_reactants

    若仅差一个 H 原子，则自动使用 1/2 H2 进行补偿（需在reactants或products中包含H2能量）

    Parameters
    ----------
    reactants : list of ase.Atoms
    products  : list of ase.Atoms

    Returns
    -------
    float : adsorption energy (eV)
    """

    def total_energy(atoms_list):
        return sum(atoms.get_potential_energy() for atoms in atoms_list)

    def count_elements(atoms_list):
        counter = Counter()
        for atoms in atoms_list:
            counter.update(atoms.get_chemical_symbols())
        return counter

    # 能量
    E_react = total_energy(reactants)
    E_prod = total_energy(products)

    # 元素统计
    count_react = count_elements(reactants)
    count_prod = count_elements(products)

    diff = count_prod - count_react  # 正值表示产物多

    # --------------------------
    # 守恒检查
    # --------------------------
    if diff == Counter():
        print("体系元素守恒。")
        return E_prod - E_react

    # 若仅差一个H
    if set(diff.keys()) == {"H"} and abs(diff["H"]) == 1:
        print("检测到仅差一个 H 原子，自动使用 1/2 H2 进行补偿。")

        # 查找H2能量
        H2_energy =  -6.859108

        if H2_energy is None:
            raise ValueError("H2分子异常，无法进行1/2 H2补偿。")

        correction = 0.5 * H2_energy

        if diff["H"] == 1:
            # 产物多一个H，相当于少了1/2 H2在反应物侧
            return E_prod - (E_react + correction)
        else:
            # 产物少一个H
            return (E_prod + correction) - E_react

    raise ValueError(f"元素不守恒，差异为: {dict(diff)}")



